// Pybind exposure for the Milestone C1/C2a sector-constraint classification/
// admission oracle and its direct pool_build_t integration
// (trimci/sector_constraint.hpp). classify_determinant_* exercises the oracle
// standalone; pool_build_with_sector_constraint_* exercises the real
// candidate-generation integration added in C2a. Neither touches Phase 0/1/2
// orchestration (iterative_workflow.cpp / fast_expansion/expansion_loop.cpp) --
// that is Milestone C2b. The plain pool_build_* bindings in bind_trimci.cpp are
// untouched; these are new, separate bindings.
#include "bind_common.hpp"

#include <limits>
#include <stdexcept>
#include <string>

#include "determinant.hpp"
#include "hamiltonian.hpp"
#include "screening.hpp"
#include "sector_constraint.hpp"

using namespace trimci_core;

namespace {

const char* label_name(SectorLabel label) {
    switch (label) {
        case SectorLabel::Protected:    return "protected";
        case SectorLabel::LasPresent:   return "las_present";
        case SectorLabel::LasAbsent:    return "las_absent";
        case SectorLabel::Dispersion:   return "dispersion";
        case SectorLabel::CtAllowed:    return "ct_allowed";
        case SectorLabel::CtDisallowed: return "ct_disallowed";
    }
    return "";
}

const char* saturation_reason_name(SaturationReason reason) {
    switch (reason) {
        case SaturationReason::None:               return "none";
        case SaturationReason::FrontierExhausted:  return "frontier_exhausted";
        case SaturationReason::ThresholdFloor:     return "threshold_floor";
        case SaturationReason::RoundLimit:         return "round_limit";
        case SaturationReason::WorkingMemoryCeiling: return "working_memory_ceiling";
    }
    return "";
}

py::dict filtered_result_to_dict(const FilteredGrowthResult& r) {
    py::dict out;
    out["n_protected"] = r.n_protected;
    out["n_recovery"] = r.n_recovery;
    out["n_aux"] = r.n_aux;
    out["n_work"] = r.n_work;
    out["recovery_target_met"] = r.recovery_target_met;
    out["doorway_limit_reached"] = r.doorway_limit_reached;
    out["saturation_reason"] = std::string(saturation_reason_name(r.saturation_reason));
    return out;
}

// Builds a SectorConstraint from plain Python-facing arguments and validates
// it -- the same "assemble a config struct from plain args inside the binding"
// convention this codebase already uses for PoolBuildParams (bind_trimci.cpp's
// bind_scalable_screening), not a stateful exposed Python class.
template<typename StorageType>
SectorConstraint<StorageType> build_and_validate_constraint(
    int n_orb,
    const std::vector<std::vector<int>>& fragment_blocks,
    const std::vector<std::pair<int, int>>& reference_sectors,
    const std::vector<std::vector<DeterminantT<StorageType>>>& fragment_support,
    const std::vector<DeterminantT<StorageType>>& protected_determinants,
    int n_ct_max,
    bool allow_las_absent_owned,
    // Milestone C2b-1 filtered-growth policy (0/max = off, matching
    // SectorConstraint's own field defaults). Irrelevant to plain
    // classify_determinant_* callers, which never set these.
    size_t recovery_target = 0,
    size_t doorway_limit = std::numeric_limits<size_t>::max(),
    size_t max_work_memory = std::numeric_limits<size_t>::max())
{
    const size_t n_frag = fragment_blocks.size();
    if (reference_sectors.size() != n_frag || fragment_support.size() != n_frag) {
        throw std::invalid_argument(
            "fragment_blocks, reference_sectors, and fragment_support must all have the same length");
    }

    SectorConstraint<StorageType> constraint;
    constraint.enabled = true;
    constraint.fragment_blocks = fragment_blocks;
    constraint.n_ct_max = n_ct_max;
    constraint.allow_las_absent_owned = allow_las_absent_owned;
    constraint.recovery_target = recovery_target;
    constraint.doorway_limit = doorway_limit;
    constraint.max_work_memory = max_work_memory;

    constraint.reference_alpha.resize(n_frag);
    constraint.reference_beta.resize(n_frag);
    for (size_t f = 0; f < n_frag; ++f) {
        constraint.reference_alpha[f] = reference_sectors[f].first;
        constraint.reference_beta[f] = reference_sectors[f].second;
    }

    constraint.fragment_support.resize(n_frag);
    for (size_t f = 0; f < n_frag; ++f) {
        for (const auto& d : fragment_support[f]) {
            constraint.fragment_support[f].insert(d);
        }
    }

    for (const auto& d : protected_determinants) {
        constraint.protected_determinants.insert(d);
    }

    validate_and_prepare_sector_constraint(constraint, n_orb);
    return constraint;
}

template<typename StorageType>
py::tuple classify_determinant_impl(
    const DeterminantT<StorageType>& det,
    int n_orb,
    const std::vector<std::vector<int>>& fragment_blocks,
    const std::vector<std::pair<int, int>>& reference_sectors,
    const std::vector<std::vector<DeterminantT<StorageType>>>& fragment_support,
    const std::vector<DeterminantT<StorageType>>& protected_determinants,
    int n_ct_max,
    bool allow_las_absent_owned,
    bool heat_bath_passed)
{
    SectorConstraint<StorageType> constraint = build_and_validate_constraint<StorageType>(
        n_orb, fragment_blocks, reference_sectors, fragment_support,
        protected_determinants, n_ct_max, allow_las_absent_owned);
    SectorClassification result = classify_determinant_t<StorageType>(det, constraint, heat_bath_passed);
    return py::make_tuple(std::string(label_name(result.label)), result.n_ct, result.admitted);
}

py::dict telemetry_to_dict(const ScreeningTelemetry& t) {
    py::dict hb, admitted, rejected;
    for (int i = 0; i < kNumSectorLabels; ++i) {
        const char* name = label_name(static_cast<SectorLabel>(i));
        hb[name] = t.hb_passing[i];
        admitted[name] = t.admitted[i];
        rejected[name] = t.rejected[i];
    }
    py::dict out;
    out["generated"] = t.generated;
    out["hb_passing"] = hb;
    out["admitted"] = admitted;
    out["rejected"] = rejected;
    return out;
}

// Exercises the actual pool_build_t integration (not just the standalone
// classifier): builds a SectorConstraint from plain args, then calls
// pool_build_t<StorageType> with it, returning
// (pool, threshold, telemetry_dict, filtered_result_dict).
// `sector_constraint_enabled=False` calls pool_build_t with a null constraint
// pointer -- the exact disabled path production code takes today.
// `recovery_target=0` (the default) keeps C2a's classic pool_map.size()
// stopping even with the constraint active for filtering; `recovery_target>0`
// is Milestone C2b-1's explicit filtered-growth opt-in, at which point
// filtered_result_dict is populated (a pre-C2b-1 caller sees an all-zero dict
// with saturation_reason="none" -- harmless, since it never checks these
// fields when it never asked for filtered growth).
template<typename StorageType>
py::tuple pool_build_with_sector_constraint_impl(
    const std::vector<DeterminantT<StorageType>>& initial_pool,
    const std::vector<double>& initial_coeff,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold,
    size_t target_size,
    HijCacheT<StorageType>& cache,
    const std::string& cache_file,
    int max_rounds,
    double threshold_decay,
    const std::vector<int>& attentive_orbitals,
    int verbosity,
    bool sector_constraint_enabled,
    const std::vector<std::vector<int>>& fragment_blocks,
    const std::vector<std::pair<int, int>>& reference_sectors,
    const std::vector<std::vector<DeterminantT<StorageType>>>& fragment_support,
    const std::vector<DeterminantT<StorageType>>& protected_determinants,
    int n_ct_max,
    bool allow_las_absent_owned,
    size_t recovery_target,
    size_t doorway_limit,
    size_t max_work_memory,
    bool strict_target_size)
{
    PoolBuildParams params;
    params.max_rounds = max_rounds;
    params.threshold_decay = threshold_decay;
    params.strict_target_size = strict_target_size;

    SectorConstraint<StorageType> constraint;
    const SectorConstraint<StorageType>* constraint_ptr = nullptr;
    if (sector_constraint_enabled) {
        constraint = build_and_validate_constraint<StorageType>(
            n_orb, fragment_blocks, reference_sectors, fragment_support,
            protected_determinants, n_ct_max, allow_las_absent_owned,
            recovery_target, doorway_limit, max_work_memory);
        constraint_ptr = &constraint;
    }

    ScreeningTelemetry telemetry;
    FilteredGrowthResult filtered_result;
    auto result = pool_build_t<StorageType>(
        initial_pool, initial_coeff, n_orb, h1, eri, threshold, target_size,
        cache, cache_file, attentive_orbitals, verbosity, params,
        nullptr, nullptr, constraint_ptr, &telemetry, &filtered_result);

    return py::make_tuple(result.first, result.second, telemetry_to_dict(telemetry),
                          filtered_result_to_dict(filtered_result));
}

template<size_t N>
void bind_scalable_sector_constraint(py::module& m, const std::string& suffix) {
    using StorageType = std::array<uint64_t, N>;
    m.def(("classify_determinant_" + suffix).c_str(),
          &classify_determinant_impl<StorageType>,
          py::arg("det"), py::arg("n_orb"), py::arg("fragment_blocks"), py::arg("reference_sectors"),
          py::arg("fragment_support"), py::arg("protected_determinants"),
          py::arg("n_ct_max") = 2, py::arg("allow_las_absent_owned") = false,
          py::arg("heat_bath_passed") = true);

    m.def(("pool_build_with_sector_constraint_" + suffix).c_str(),
          &pool_build_with_sector_constraint_impl<StorageType>,
          py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"), py::arg("h1"), py::arg("eri"),
          py::arg("threshold"), py::arg("target_size"), py::arg("cache"), py::arg("cache_file"),
          py::arg("max_rounds") = 2, py::arg("threshold_decay") = 0.5,
          py::arg("attentive_orbitals") = std::vector<int>{}, py::arg("verbosity") = 0,
          py::arg("sector_constraint_enabled") = false,
          py::arg("fragment_blocks") = std::vector<std::vector<int>>{},
          py::arg("reference_sectors") = std::vector<std::pair<int, int>>{},
          py::arg("fragment_support") = std::vector<std::vector<DeterminantT<StorageType>>>{},
          py::arg("protected_determinants") = std::vector<DeterminantT<StorageType>>{},
          py::arg("n_ct_max") = 2, py::arg("allow_las_absent_owned") = false,
          py::arg("recovery_target") = size_t(0),
          py::arg("doorway_limit") = std::numeric_limits<size_t>::max(),
          py::arg("max_work_memory") = std::numeric_limits<size_t>::max(),
          py::arg("strict_target_size") = false);
}

} // namespace

void bind_sector_constraint(py::module& m) {
    m.def("classify_determinant_64", &classify_determinant_impl<uint64_t>,
          py::arg("det"), py::arg("n_orb"), py::arg("fragment_blocks"), py::arg("reference_sectors"),
          py::arg("fragment_support"), py::arg("protected_determinants"),
          py::arg("n_ct_max") = 2, py::arg("allow_las_absent_owned") = false,
          py::arg("heat_bath_passed") = true);

    m.def("pool_build_with_sector_constraint_64",
          &pool_build_with_sector_constraint_impl<uint64_t>,
          py::arg("initial_pool"), py::arg("initial_coeff"), py::arg("n_orb"), py::arg("h1"), py::arg("eri"),
          py::arg("threshold"), py::arg("target_size"), py::arg("cache"), py::arg("cache_file"),
          py::arg("max_rounds") = 2, py::arg("threshold_decay") = 0.5,
          py::arg("attentive_orbitals") = std::vector<int>{}, py::arg("verbosity") = 0,
          py::arg("sector_constraint_enabled") = false,
          py::arg("fragment_blocks") = std::vector<std::vector<int>>{},
          py::arg("reference_sectors") = std::vector<std::pair<int, int>>{},
          py::arg("fragment_support") = std::vector<std::vector<DeterminantT<uint64_t>>>{},
          py::arg("protected_determinants") = std::vector<DeterminantT<uint64_t>>{},
          py::arg("n_ct_max") = 2, py::arg("allow_las_absent_owned") = false,
          py::arg("recovery_target") = size_t(0),
          py::arg("doorway_limit") = std::numeric_limits<size_t>::max(),
          py::arg("max_work_memory") = std::numeric_limits<size_t>::max(),
          py::arg("strict_target_size") = false);

    bind_scalable_sector_constraint<2>(m, "128");
    bind_scalable_sector_constraint<3>(m, "192");
    bind_scalable_sector_constraint<4>(m, "256");
    bind_scalable_sector_constraint<5>(m, "320");
    bind_scalable_sector_constraint<6>(m, "384");
    bind_scalable_sector_constraint<7>(m, "448");
    bind_scalable_sector_constraint<8>(m, "512");
}
