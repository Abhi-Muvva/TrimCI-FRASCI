#pragma once

// Shared Python-facing helpers for the filter-aware recovery feature
// (Milestone C2b-2). Lives in its own header because three binding TUs now
// need the same conversions: bind_sector_constraint.cpp (classifier +
// pool_build), bind_workflow.cpp (Phase 0) and bind_fast_expansion.cpp
// (Phase 1/2).

#include "bind_common.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "determinant.hpp"
#include "sector_constraint.hpp"

namespace trimci_bind {

using namespace trimci_core;

inline const char* label_name(SectorLabel label) {
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

inline const char* saturation_reason_name(SaturationReason reason) {
    switch (reason) {
        case SaturationReason::None:                 return "none";
        case SaturationReason::FrontierExhausted:    return "frontier_exhausted";
        case SaturationReason::ThresholdFloor:       return "threshold_floor";
        case SaturationReason::RoundLimit:           return "round_limit";
        case SaturationReason::WorkingMemoryCeiling: return "working_memory_ceiling";
    }
    return "";
}

inline py::dict filtered_result_to_dict(const FilteredGrowthResult& r) {
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

inline py::dict telemetry_to_dict(const ScreeningTelemetry& t) {
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

inline py::dict stage_to_dict(const FilteredStageTelemetry& s) {
    py::dict out;
    out["phase"] = s.phase;
    out["index"] = s.index;
    out["restart"] = s.restart;
    out["events"] = telemetry_to_dict(s.events);
    out["pool"] = filtered_result_to_dict(s.pool);
    out["state"] = filtered_result_to_dict(s.state);
    out["requested_recovery_target"] = s.requested_recovery_target;
    out["target_met"] = s.target_met;
    out["doorway_limit_reached"] = s.pool.doorway_limit_reached;
    out["saturation_reason"] = std::string(saturation_reason_name(s.pool.saturation_reason));
    out["termination_reason"] = s.termination_reason;
    return out;
}

inline py::list workflow_telemetry_to_list(const FilteredWorkflowTelemetry& t) {
    py::list out;
    for (const auto& s : t.stages) out.append(stage_to_dict(s));
    return out;
}

// Builds and validates a SectorConstraint from plain Python-facing arguments.
// Returned by shared_ptr so one immutable instance can be shared by Phase 0,
// Phase 1, and Phase 2 with no copy, and so pybind keeps it alive for as long
// as any config or in-flight native call references it -- the lifetime-safe
// alternative to handing C++ a raw pointer across a GIL release.
template<typename StorageType>
std::shared_ptr<SectorConstraint<StorageType>> make_sector_constraint(
    int n_orb,
    const std::vector<std::vector<int>>& fragment_blocks,
    const std::vector<std::pair<int, int>>& reference_sectors,
    const std::vector<std::vector<DeterminantT<StorageType>>>& fragment_support,
    const std::vector<DeterminantT<StorageType>>& protected_determinants,
    int n_ct_max,
    bool allow_las_absent_owned)
{
    const size_t n_frag = fragment_blocks.size();
    if (reference_sectors.size() != n_frag || fragment_support.size() != n_frag) {
        throw std::invalid_argument(
            "fragment_blocks, reference_sectors, and fragment_support must all have the same length");
    }

    auto constraint = std::make_shared<SectorConstraint<StorageType>>();
    constraint->enabled = true;
    constraint->fragment_blocks = fragment_blocks;
    constraint->n_ct_max = n_ct_max;
    constraint->allow_las_absent_owned = allow_las_absent_owned;

    constraint->reference_alpha.resize(n_frag);
    constraint->reference_beta.resize(n_frag);
    for (size_t f = 0; f < n_frag; ++f) {
        constraint->reference_alpha[f] = reference_sectors[f].first;
        constraint->reference_beta[f] = reference_sectors[f].second;
    }

    constraint->fragment_support.resize(n_frag);
    for (size_t f = 0; f < n_frag; ++f) {
        for (const auto& d : fragment_support[f]) constraint->fragment_support[f].insert(d);
    }
    for (const auto& d : protected_determinants) constraint->protected_determinants.insert(d);

    validate_and_prepare_sector_constraint(*constraint, n_orb);
    return constraint;
}

// Opaque, already-validated, immutable handle plus the helpers that operate on
// one. Exposing a handle (rather than rebuilding the constraint from plain
// args at every entry point, as C1/C2a did) matters now that Phase 0, Phase 1,
// Phase 2, and the restart-supplement screener all need the same constraint:
// validation and the fragment-support tables are paid once.
template<typename StorageType>
void bind_sector_constraint_handle(py::module& m, const std::string& suffix) {
    using C = SectorConstraint<StorageType>;
    py::class_<C, std::shared_ptr<C>>(m, ("SectorConstraint" + suffix).c_str())
        .def_property_readonly("n_fragments",
            [](const C& c) { return c.fragment_blocks.size(); })
        .def_property_readonly("n_protected",
            [](const C& c) { return c.protected_determinants.size(); })
        .def_readonly("n_ct_max", &C::n_ct_max)
        .def_readonly("allow_las_absent_owned", &C::allow_las_absent_owned)
        .def_readonly("prepared", &C::prepared);

    m.def(("make_sector_constraint_" + suffix).c_str(),
          &make_sector_constraint<StorageType>,
          py::arg("n_orb"), py::arg("fragment_blocks"), py::arg("reference_sectors"),
          py::arg("fragment_support"), py::arg("protected_determinants"),
          py::arg("n_ct_max") = 2, py::arg("allow_las_absent_owned") = false,
          "Build and validate a SectorConstraint handle (reusable across phases).");

    // Bulk classifier for candidate initial/supplement determinants. This is
    // what a Phase-0 restart generator uses to DROP a randomly generated
    // supplement that the admission rule would forbid -- deliberately
    // different from the strict validation applied to a supplied/loaded state,
    // which fails loudly instead.
    m.def(("screen_dets_" + suffix).c_str(),
          [](const C& constraint, const std::vector<DeterminantT<StorageType>>& dets) {
              py::list labels, admitted;
              for (const auto& d : dets) {
                  SectorClassification r = classify_determinant_t(d, constraint, true);
                  labels.append(std::string(label_name(r.label)));
                  admitted.append(r.admitted);
              }
              return py::make_tuple(labels, admitted);
          },
          py::arg("constraint"), py::arg("dets"),
          "Classify many determinants against one prepared constraint -> (labels, admitted).");

    m.def(("compose_and_validate_" + suffix).c_str(),
          [](const C& constraint, const std::vector<DeterminantT<StorageType>>& dets,
             size_t recovery_target, const std::string& context) {
              return filtered_result_to_dict(
                  compose_and_validate_t(dets, constraint, recovery_target, context));
          },
          py::arg("constraint"), py::arg("dets"),
          py::arg("recovery_target") = size_t(0),
          py::arg("context") = std::string("compose_and_validate"),
          "Reclassify a determinant set into P_K / R / O; raises on a forbidden "
          "determinant or a missing protected determinant.");
}

}  // namespace trimci_bind
