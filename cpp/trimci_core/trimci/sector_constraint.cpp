#include "sector_constraint.hpp"

#include <cstdlib>

namespace trimci_core {

template<typename StorageType>
SectorClassification classify_determinant_t(
    const DeterminantT<StorageType>& det,
    const SectorConstraint<StorageType>& constraint,
    bool heat_bath_passed)
{
    if (constraint.protected_determinants.count(det)) {
        return SectorClassification{SectorLabel::Protected, 0, true};
    }

    const size_t n_frag = constraint.fragment_blocks.size();
    bool owned = true;
    int n_ct_sum = 0;  // half-sum of |fragment total-electron deviation|; n_ct = n_ct_sum / 2

    for (size_t f = 0; f < n_frag; ++f) {
        const auto& block = constraint.fragment_blocks[f];
        int na = 0, nb = 0;
        for (int orb : block) {
            if (BitOps<StorageType>::get_bit(det.alpha, orb)) ++na;
            if (BitOps<StorageType>::get_bit(det.beta, orb)) ++nb;
        }
        if (na != constraint.reference_alpha[f] || nb != constraint.reference_beta[f]) {
            owned = false;
        }
        n_ct_sum += std::abs((na + nb) - (constraint.reference_alpha[f] + constraint.reference_beta[f]));
    }

    if (owned) {
        bool las_present = true;
        for (size_t f = 0; f < n_frag; ++f) {
            const auto& block = constraint.fragment_blocks[f];
            uint64_t local_alpha = 0, local_beta = 0;
            for (size_t local = 0; local < block.size(); ++local) {
                int orb = block[local];
                if (BitOps<StorageType>::get_bit(det.alpha, orb)) local_alpha |= (uint64_t(1) << local);
                if (BitOps<StorageType>::get_bit(det.beta, orb)) local_beta |= (uint64_t(1) << local);
            }
            uint64_t key = (local_alpha << block.size()) | local_beta;
            if (!constraint.fragment_support[f].count(key)) {
                las_present = false;
                break;
            }
        }
        if (las_present) {
            return SectorClassification{SectorLabel::LasPresent, 0, false};
        }
        bool admitted = constraint.allow_las_absent_owned && heat_bath_passed;
        return SectorClassification{SectorLabel::LasAbsent, 0, admitted};
    }

    int n_ct = n_ct_sum / 2;
    if (n_ct == 0) {
        return SectorClassification{SectorLabel::Dispersion, 0, heat_bath_passed};
    }
    if (n_ct <= constraint.n_ct_max) {
        return SectorClassification{SectorLabel::CtAllowed, n_ct, heat_bath_passed};
    }
    return SectorClassification{SectorLabel::CtDisallowed, n_ct, false};
}

// Explicit instantiations: every determinant storage type pool_build_t itself
// already supports (screening.cpp's own instantiation list), so the sector
// constraint has no 64-orbital-only special path.
template SectorClassification classify_determinant_t<uint64_t>(
    const DeterminantT<uint64_t>&, const SectorConstraint<uint64_t>&, bool);
template SectorClassification classify_determinant_t<std::array<uint64_t, 2>>(
    const DeterminantT<std::array<uint64_t, 2>>&, const SectorConstraint<std::array<uint64_t, 2>>&, bool);
template SectorClassification classify_determinant_t<std::array<uint64_t, 3>>(
    const DeterminantT<std::array<uint64_t, 3>>&, const SectorConstraint<std::array<uint64_t, 3>>&, bool);
template SectorClassification classify_determinant_t<std::array<uint64_t, 4>>(
    const DeterminantT<std::array<uint64_t, 4>>&, const SectorConstraint<std::array<uint64_t, 4>>&, bool);
template SectorClassification classify_determinant_t<std::array<uint64_t, 5>>(
    const DeterminantT<std::array<uint64_t, 5>>&, const SectorConstraint<std::array<uint64_t, 5>>&, bool);
template SectorClassification classify_determinant_t<std::array<uint64_t, 6>>(
    const DeterminantT<std::array<uint64_t, 6>>&, const SectorConstraint<std::array<uint64_t, 6>>&, bool);
template SectorClassification classify_determinant_t<std::array<uint64_t, 7>>(
    const DeterminantT<std::array<uint64_t, 7>>&, const SectorConstraint<std::array<uint64_t, 7>>&, bool);
template SectorClassification classify_determinant_t<std::array<uint64_t, 8>>(
    const DeterminantT<std::array<uint64_t, 8>>&, const SectorConstraint<std::array<uint64_t, 8>>&, bool);

} // namespace trimci_core
