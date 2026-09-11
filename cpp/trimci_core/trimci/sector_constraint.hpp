#pragma once

// Filter-aware phased TrimCI interfragment recovery: native sector constraint,
// classification/admission oracle, and screening-loop integration (Milestones
// C1, C2a, C2b-1).
//
// C2a wired classification into pool_build_t's two direct candidate-generation
// paths (screening.cpp). C2b-1 adds unique recovery-pool composition
// (N_protected/N_recovery/N_aux/N_work), recovery-target stopping, a doorway
// insertion limit, and category-aware strict truncation -- all still inside
// direct pool_build_t only. Phase 0/1/2 orchestration (iterative_workflow.cpp,
// fast_expansion/expansion_loop.cpp), ExpansionConfig, run_trim_t, and
// protected-seed survival across TRIM/core reselection and phase handoffs are
// all still untouched -- that is Milestone C2b-2.
//
// Mirrors frasci.recovery._state.classify_recovery() exactly (Python oracle,
// LASSCF_TRIMCI_COO/frasci/recovery/_state.py) -- same six labels, same truth
// table.

#include <array>
#include <cstdint>
#include <limits>
#include <unordered_map>
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

    // Milestone C2b-1 filtered-growth policy. Explicit opt-in: recovery_target
    // == 0 (the default) means "filtered growth stopping is not requested" --
    // pool_build_t keeps its original pool_map.size()-based stopping even with
    // this constraint active for classification/filtering (exactly C2a's
    // behavior, still available and still tested). recovery_target > 0 is the
    // named "filtered-pool policy" that switches pool_build_t to stop on
    // unique N_recovery, apply the doorway_limit, enforce max_work_memory, and
    // use category-aware strict truncation -- never on total pool size.
    size_t recovery_target = 0;
    // Maximum unique admitted LAS-absent (doorway, O) pool members. Reaching
    // this blocks further doorway insertions but never blocks recovery (R) or
    // protected (P_K) insertions, and is not itself a saturation/stop reason.
    size_t doorway_limit = std::numeric_limits<size_t>::max();
    // Independent ceiling on unique total pool members (N_work), regardless of
    // recovery_target. If this is reached before recovery_target is met,
    // growth stops with SaturationReason::WorkingMemoryCeiling.
    size_t max_work_memory = std::numeric_limits<size_t>::max();

    // Derived: one full-width orbital mask per fragment (bit set at every
    // orbital in fragment_blocks[f]), computed by validate_and_prepare_sector_
    // constraint. classify_determinant_t / screen_candidate_t require
    // `prepared == true`.
    std::vector<StorageType> fragment_mask;
    bool prepared = false;
};

// Why pool_build_t's growth stopped without reaching recovery_target. Only
// meaningful when FilteredGrowthResult::recovery_target_met is false --
// deliberately does NOT include a "doorway limit" value: reaching the doorway
// limit alone never stops growth (R must still be able to enter), so it is
// reported as a separate FilteredGrowthResult::doorway_limit_reached flag
// instead of a mutually-exclusive terminal reason.
enum class SaturationReason : int {
    None = 0,
    FrontierExhausted = 1,
    ThresholdFloor = 2,
    RoundLimit = 3,
    WorkingMemoryCeiling = 4,
};

// Unique pool composition (Milestone C2b-1) -- NOT the same thing as
// ScreeningTelemetry's event counts above. A determinant counts here exactly
// once, when pool_map.try_emplace (or initial-pool seeding) actually inserts
// it; ScreeningTelemetry can and often will show more events than these
// counts show insertions.
struct FilteredGrowthResult {
    uint64_t n_protected = 0;  // unique P_K members
    uint64_t n_recovery = 0;   // unique dispersion + ct_allowed members (R)
    uint64_t n_aux = 0;        // unique admitted las_absent doorway members (O)
    uint64_t n_work = 0;       // unique total pool members (P_K + R + O)
    bool recovery_target_met = false;
    bool doorway_limit_reached = false;
    SaturationReason saturation_reason = SaturationReason::None;
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
//     entirely of fragment-local determinants already in fragment_support);
//   - (Milestone C2b-1) recovery_target > 0 (filtered-growth policy active)
//     with max_work_memory < protected_determinants.size() + recovery_target
//     -- the working capacity cannot possibly hold P_K plus the requested
//     recovery population, so this fails at configuration time rather than
//     after a doomed growth run.
// Must be called exactly once before classify_determinant_t / screen_candidate_t
// are used on a given constraint; sets constraint.prepared = true on success.
template<typename StorageType>
void validate_and_prepare_sector_constraint(SectorConstraint<StorageType>& constraint, int n_orb);

// Milestone C2b-1: classifies every member of a caller-supplied initial pool
// exactly once (as if heat_bath_passed=true -- seeds are not subject to
// heat-bath admission) and returns a label per determinant, in input order.
// Throws std::invalid_argument if any NON-protected member classifies as
// LasPresent or CtDisallowed: an initial pool is not itself subject to the
// candidate-generation admission rule, but it must not silently contain
// determinants that rule would never have produced. Requires
// constraint.prepared.
template<typename StorageType>
std::vector<SectorLabel> classify_initial_pool_t(
    const std::vector<DeterminantT<StorageType>>& initial_pool,
    const SectorConstraint<StorageType>& constraint);

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
