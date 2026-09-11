// Pybind exposure for the Milestone C1 sector-constraint classification/admission
// oracle (trimci/sector_constraint.hpp). Nothing here touches pool_build_t,
// process_parent_worker_t, or PoolBuildParams -- these bindings exist so the
// classifier can be exercised and cross-checked directly from Python against
// frasci.recovery._state.classify_recovery(), independent of the actual
// candidate-generation loop (Milestone C2's job).
#include "bind_common.hpp"

#include <stdexcept>
#include <string>

#include "determinant.hpp"
#include "sector_constraint.hpp"

using namespace trimci_core;

namespace {

template<typename StorageType>
py::tuple classify_determinant_impl(
    const DeterminantT<StorageType>& det,
    const std::vector<std::vector<int>>& fragment_blocks,
    const std::vector<std::pair<int, int>>& reference_sectors,
    const std::vector<std::vector<std::pair<uint64_t, uint64_t>>>& fragment_support,
    const std::vector<DeterminantT<StorageType>>& protected_determinants,
    int n_ct_max,
    bool allow_las_absent_owned,
    bool heat_bath_passed)
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

    constraint.reference_alpha.resize(n_frag);
    constraint.reference_beta.resize(n_frag);
    for (size_t f = 0; f < n_frag; ++f) {
        constraint.reference_alpha[f] = reference_sectors[f].first;
        constraint.reference_beta[f] = reference_sectors[f].second;
    }

    constraint.fragment_support.resize(n_frag);
    for (size_t f = 0; f < n_frag; ++f) {
        const size_t block_size = fragment_blocks[f].size();
        for (const auto& local_pair : fragment_support[f]) {
            uint64_t key = (local_pair.first << block_size) | local_pair.second;
            constraint.fragment_support[f].insert(key);
        }
    }

    for (const auto& d : protected_determinants) {
        constraint.protected_determinants.insert(d);
    }

    SectorClassification result = classify_determinant_t<StorageType>(det, constraint, heat_bath_passed);

    const char* label_str = "";
    switch (result.label) {
        case SectorLabel::Protected:     label_str = "protected"; break;
        case SectorLabel::LasPresent:    label_str = "las_present"; break;
        case SectorLabel::LasAbsent:     label_str = "las_absent"; break;
        case SectorLabel::Dispersion:    label_str = "dispersion"; break;
        case SectorLabel::CtAllowed:     label_str = "ct_allowed"; break;
        case SectorLabel::CtDisallowed:  label_str = "ct_disallowed"; break;
    }
    return py::make_tuple(std::string(label_str), result.n_ct, result.admitted);
}

template<size_t N>
void bind_scalable_sector_constraint(py::module& m, const std::string& suffix) {
    m.def(("classify_determinant_" + suffix).c_str(),
          &classify_determinant_impl<std::array<uint64_t, N>>,
          py::arg("det"), py::arg("fragment_blocks"), py::arg("reference_sectors"),
          py::arg("fragment_support"), py::arg("protected_determinants"),
          py::arg("n_ct_max") = 2, py::arg("allow_las_absent_owned") = false,
          py::arg("heat_bath_passed") = true);
}

} // namespace

void bind_sector_constraint(py::module& m) {
    m.def("classify_determinant_64", &classify_determinant_impl<uint64_t>,
          py::arg("det"), py::arg("fragment_blocks"), py::arg("reference_sectors"),
          py::arg("fragment_support"), py::arg("protected_determinants"),
          py::arg("n_ct_max") = 2, py::arg("allow_las_absent_owned") = false,
          py::arg("heat_bath_passed") = true);

    bind_scalable_sector_constraint<2>(m, "128");
    bind_scalable_sector_constraint<3>(m, "192");
    bind_scalable_sector_constraint<4>(m, "256");
    bind_scalable_sector_constraint<5>(m, "320");
    bind_scalable_sector_constraint<6>(m, "384");
    bind_scalable_sector_constraint<7>(m, "448");
    bind_scalable_sector_constraint<8>(m, "512");
}
