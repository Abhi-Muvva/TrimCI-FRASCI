#pragma once

// Filter-aware phased TrimCI interfragment recovery: native sector constraint,
// classification/admission oracle, and screening-loop integration (Milestones
// C1 + C2a).
//
// C2a wires this into pool_build_t's two direct candidate-generation paths
// (screening.cpp) only. Phase 0/1/2 orchestration (iterative_workflow.cpp,
// fast_expansion/expansion_loop.cpp), phase stopping logic, protected-seed
// survival across truncation/restarts, and doorway/N_recovery accounting are
// all still untouched -- that is Milestone C2b.
//
// Mirrors frasci.recovery._state.classify_recovery() exactly (Python oracle,
// LASSCF_TRIMCI_COO/frasci/recovery/_state.py) -- same six labels, same truth
// table.

#include <array>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "determinant.hpp"

namespace trimci_core {

enum class SectorLabel : int {
    Protected = 0,
    LasPresent = 1,
    LasAbsent = 2,
    Dispersion = 3,
    CtAllowed = 4,
    CtDisallowed = 5,
};
constexpr int kNumSectorLabels = 6;

struct SectorClassification {
    SectorLabel label;
    int n_ct;
    bool admitted;
};

// Event telemetry for the screening loop (Milestone C2a scope). These are EVENT
// counts, not unique-pool-membership counts: a determinant independently
// (re)generated as a candidate from several different parents in the same round
// increments these several times, even though pool_map (the actual candidate
// pool, unaffected by this struct) only ever holds it once. Unique N_recovery /
// doorway accounting is Milestone C2b.
//
// `generated` is the raw process_parent_worker_t output count, before the
// heat-bath threshold test, and is NOT broken down by class (a rejected-by-
// heat-bath child is never classified at all, matching "call the screening
// helper after the existing heat-bath condition passes"). `hb_passing`,
// `admitted`, and `rejected` are all indexed by SectorLabel and only populated
// for children that already passed heat-bath.
struct ScreeningTelemetry {
    uint64_t generated = 0;
    std::array<uint64_t, kNumSectorLabels> hb_passing{};
    std::array<uint64_t, kNumSectorLabels> admitted{};
    std::array<uint64_t, kNumSectorLabels> rejected{};

    void reset() {
        generated = 0;
        hb_passing.fill(0);
        admitted.fill(0);
        rejected.fill(0);
    }

    void merge(const ScreeningTelemetry& other) {
        generated += other.generated;
        for (int i = 0; i < kNumSectorLabels; ++i) {
            hb_passing[i] += other.hb_passing[i];
            admitted[i] += other.admitted[i];
            rejected[i] += other.rejected[i];
        }
    }
};

// Data-only, immutable-after-validate_and_prepare sector constraint. No Python
// callback, no global mutable state: every field here is plain data the
// classifier reads. Passed to pool_build_t as a nullable typed pointer -- never
// a bool+void* pair -- so disabled/absent is simply `nullptr` and the templated
// caller needs no cast.
template<typename StorageType>
struct SectorConstraint {
    bool enabled = false;

    // fragment_blocks[f] = GLOBAL orbital indices owned by fragment f. Need not
    // be contiguous or sorted, but must be disjoint and in [0, n_orb) --
    // enforced by validate_and_prepare_sector_constraint, not here.
    std::vector<std::vector<int>> fragment_blocks;

    // LAS reference (n_alpha, n_beta) per fragment, same order as fragment_blocks.
    std::vector<int> reference_alpha;
    std::vector<int> reference_beta;

    // fragment_support[f]: one entry per fragment-local determinant present in
    // fragment f's stored LAS expansion, EMBEDDED in the global orbital frame
    // (every bit outside fragment_blocks[f] must be zero -- validated, not
    // assumed). Deliberately NOT a packed integer key: reusing
    // DeterminantT<StorageType>'s own storage and hash specialization is
    // arbitrary-width-safe by construction (no shift-overflow ceiling to track,
    // unlike a hand-packed `(local_alpha << block_size) | local_beta` key,
    // which silently produces undefined behavior once block_size approaches
    // the word width). This replaces C1's packed-uint64_t representation.
    std::vector<std::unordered_set<DeterminantT<StorageType>>> fragment_support;

    // Protected seed P_K: global determinant identity, always admitted.
    // validate_and_prepare_sector_constraint requires every entry here to
    // already be LAS-present (P_K subset-of P_LAS, per the design's own
    // definition) -- this is a genuine precondition, not merely convenient.
    std::unordered_set<DeterminantT<StorageType>> protected_determinants;

    int n_ct_max = 2;
    bool allow_las_absent_owned = false;

    // Derived: one full-width orbital mask per fragment (bit set at every
    // orbital in fragment_blocks[f]), computed by validate_and_prepare_sector_
    // constraint. classify_determinant_t / screen_candidate_t require
    // `prepared == true`.
    std::vector<StorageType> fragment_mask;
    bool prepared = false;
};

// Cross-checks every field for internal consistency and precomputes
// fragment_mask. Throws std::invalid_argument (with a specific message) on the
// first violation found:
//   - fragment_blocks / reference_alpha / reference_beta / fragment_support
//     length mismatch;
//   - an orbital index outside [0, n_orb) or claimed by more than one block;
//   - a reference sector inconsistent with its own block size;
//   - n_ct_max < 0;
//   - a fragment_support entry with bits outside its own block, or whose
//     occupation does not match that fragment's own reference sector;
//   - a protected determinant that is not LAS-present (i.e. not composed
//     entirely of fragment-local determinants already in fragment_support).
// Must be called exactly once before classify_determinant_t / screen_candidate_t
// are used on a given constraint; sets constraint.prepared = true on success.
template<typename StorageType>
void validate_and_prepare_sector_constraint(SectorConstraint<StorageType>& constraint, int n_orb);

// Classify one candidate determinant and decide admission, per
// docs/methods/filter-aware-phased-trimci-interfragment-recovery.md sec 3.3's
// truth table:
//
//   protected            -> always admitted
//   las_present (\ P_K)  -> always rejected (redundant; restored via full P_LAS later)
//   las_absent           -> admitted iff allow_las_absent_owned && heat_bath_passed
//   dispersion           -> admitted iff heat_bath_passed
//   ct_allowed           -> admitted iff heat_bath_passed
//   ct_disallowed        -> always rejected
//
// `constraint.enabled`/`prepared` are not consulted here: production call
// sites are responsible for only calling this when both are true (screen_
// candidate_t below is the one shared call site that does so).
template<typename StorageType>
SectorClassification classify_determinant_t(
    const DeterminantT<StorageType>& det,
    const SectorConstraint<StorageType>& constraint,
    bool heat_bath_passed);

// Shared screening-loop helper (Milestone C2a): the SINGLE call site both the
// heat_bath/uniform path and the PT2/MI path in pool_build_t use, so the
// admission rule is never duplicated per excitation-type loop. `child` must
// already have passed the caller's existing heat-bath threshold test --
// heat_bath_passed=true is implied, not re-derived here.
template<typename StorageType>
inline SectorClassification screen_candidate_t(
    const DeterminantT<StorageType>& child,
    const SectorConstraint<StorageType>& constraint)
{
    return classify_determinant_t(child, constraint, /*heat_bath_passed=*/true);
}

} // namespace trimci_core
