#pragma once

// Filter-aware phased TrimCI interfragment recovery: native sector constraint
// and classification/admission oracle (Milestone C1).
//
// This header is intentionally standalone: nothing in the existing pool_build_t /
// process_parent_worker_t path references it yet. Wiring SectorConstraint into
// PoolBuildParams and the actual candidate-generation loop is Milestone C2's job.
// A disabled-by-default flag is included now so C2 can add the field without a
// second migration, but its only consumer today is classify_determinant_t itself.
//
// Mirrors frasci.recovery._state.classify_recovery() exactly (Python oracle,
// LASSCF_TRIMCI_COO/frasci/recovery/_state.py) -- same six labels, same truth
// table. heat_bath_passed is an explicit input, not computed here: this data
// model does not reproduce TrimCI's own heat-bath scoring, matching the
// design's Gate 1 scope (classification correctness, not selector integration).

#include <cstdint>
#include <unordered_set>
#include <vector>

#include "determinant.hpp"

namespace trimci_core {

enum class SectorLabel {
    Protected,
    LasPresent,
    LasAbsent,
    Dispersion,
    CtAllowed,
    CtDisallowed,
};

struct SectorClassification {
    SectorLabel label;
    int n_ct;
    bool admitted;
};

// Data-only, immutable-during-a-solve sector constraint. No Python callback,
// no global mutable state: every field here is plain data the classifier reads.
template<typename StorageType>
struct SectorConstraint {
    bool enabled = false;

    // fragment_blocks[f] = GLOBAL orbital indices owned by fragment f. Need not
    // be contiguous or sorted; classify_determinant_t only tests bit membership.
    std::vector<std::vector<int>> fragment_blocks;

    // LAS reference (n_alpha, n_beta) per fragment, same order as fragment_blocks.
    std::vector<int> reference_alpha;
    std::vector<int> reference_beta;

    // Per-fragment LAS-support lookup: fragment_support[f] contains one packed
    // key per fragment-local determinant present in that fragment's stored LAS
    // expansion. key = (local_alpha << |block_f|) | local_beta, where local_alpha/
    // local_beta pack bit `local` for block_f[local] (same convention as
    // frasci.recovery._state._fragment_lookup_table's local_key). Assumes
    // |block_f| <= 62 (matches the Python oracle's own "well under int64 range"
    // assumption; not re-validated here).
    std::vector<std::unordered_set<uint64_t>> fragment_support;

    // Protected seed P_K: global determinant identity, always admitted.
    std::unordered_set<DeterminantT<StorageType>> protected_determinants;

    int n_ct_max = 2;
    bool allow_las_absent_owned = false;
};

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
// `constraint.enabled` is not consulted here: a disabled constraint is simply
// never called by production code (C2's job to guard that call site). This
// function is the classification oracle itself, exercised directly by tests
// and, from C2 onward, by the real screening loop.
template<typename StorageType>
SectorClassification classify_determinant_t(
    const DeterminantT<StorageType>& det,
    const SectorConstraint<StorageType>& constraint,
    bool heat_bath_passed
);

} // namespace trimci_core
