#include "sector_constraint.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace trimci_core {

namespace {

template<typename StorageType>
StorageType bitwise_and(const StorageType& a, const StorageType& b) {
    if constexpr (std::is_same_v<StorageType, uint64_t>) {
        return a & b;
    } else {
        StorageType out{};
        for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] & b[i];
        return out;
    }
}

// Fragment-support lookup only (does NOT check the fragment sector itself --
// callers that have not already established `owned` must use is_las_present).
template<typename StorageType>
bool fragment_support_lookup_all(
    const DeterminantT<StorageType>& det, const SectorConstraint<StorageType>& c)
{
    for (size_t f = 0; f < c.fragment_blocks.size(); ++f) {
        StorageType masked_alpha = bitwise_and(det.alpha, c.fragment_mask[f]);
        StorageType masked_beta = bitwise_and(det.beta, c.fragment_mask[f]);
        if (!c.fragment_support[f].count(DeterminantT<StorageType>(masked_alpha, masked_beta))) {
            return false;
        }
    }
    return true;
}

// Full LAS-present test: same-sector in every fragment AND every fragment's
// local restriction is in that fragment's stored support. Used by validation
// (protected determinants) where "owned" has not already been established, so
// it cannot reuse the classify_determinant_t hot path (which special-cases
// `protected_determinants` first).
template<typename StorageType>
bool is_las_present(const DeterminantT<StorageType>& det, const SectorConstraint<StorageType>& c) {
    for (size_t f = 0; f < c.fragment_blocks.size(); ++f) {
        const auto& block = c.fragment_blocks[f];
        int na = 0, nb = 0;
        for (int orb : block) {
            if (BitOps<StorageType>::get_bit(det.alpha, orb)) ++na;
            if (BitOps<StorageType>::get_bit(det.beta, orb)) ++nb;
        }
        if (na != c.reference_alpha[f] || nb != c.reference_beta[f]) return false;
    }
    return fragment_support_lookup_all(det, c);
}

} // namespace

template<typename StorageType>
void validate_and_prepare_sector_constraint(SectorConstraint<StorageType>& constraint, int n_orb) {
    const size_t n_frag = constraint.fragment_blocks.size();
    if (constraint.reference_alpha.size() != n_frag
        || constraint.reference_beta.size() != n_frag
        || constraint.fragment_support.size() != n_frag) {
        throw std::invalid_argument(
            "SectorConstraint: fragment_blocks, reference_alpha, reference_beta, and "
            "fragment_support must all have the same length");
    }

    std::vector<bool> orbital_claimed(static_cast<size_t>(n_orb), false);
    constraint.fragment_mask.assign(n_frag, StorageType{});

    for (size_t f = 0; f < n_frag; ++f) {
        const auto& block = constraint.fragment_blocks[f];
        StorageType mask{};
        for (int orb : block) {
            if (orb < 0 || orb >= n_orb) {
                throw std::invalid_argument(
                    "SectorConstraint: fragment " + std::to_string(f) + " orbital "
                    + std::to_string(orb) + " is out of range [0, " + std::to_string(n_orb) + ")");
            }
            if (orbital_claimed[static_cast<size_t>(orb)]) {
                throw std::invalid_argument(
                    "SectorConstraint: orbital " + std::to_string(orb)
                    + " is claimed by more than one fragment block");
            }
            orbital_claimed[static_cast<size_t>(orb)] = true;
            BitOps<StorageType>::set_bit(mask, orb);
        }
        constraint.fragment_mask[f] = mask;

        const int ra = constraint.reference_alpha[f];
        const int rb = constraint.reference_beta[f];
        const int block_size = static_cast<int>(block.size());
        if (ra < 0 || rb < 0 || ra > block_size || rb > block_size) {
            throw std::invalid_argument(
                "SectorConstraint: fragment " + std::to_string(f) + " reference sector ("
                + std::to_string(ra) + "," + std::to_string(rb) + ") is inconsistent with its "
                + std::to_string(block_size) + "-orbital block");
        }

        for (const auto& local_det : constraint.fragment_support[f]) {
            const size_t total_a = local_det.getOccupiedAlpha().size();
            const size_t total_b = local_det.getOccupiedBeta().size();
            const StorageType masked_alpha = bitwise_and(local_det.alpha, mask);
            const StorageType masked_beta = bitwise_and(local_det.beta, mask);
            const size_t in_mask_a =
                DeterminantT<StorageType>(masked_alpha, StorageType{}).getOccupiedAlpha().size();
            const size_t in_mask_b =
                DeterminantT<StorageType>(StorageType{}, masked_beta).getOccupiedBeta().size();
            if (in_mask_a != total_a || in_mask_b != total_b) {
                throw std::invalid_argument(
                    "SectorConstraint: fragment " + std::to_string(f)
                    + " support determinant has bits outside its own orbital block");
            }
            if (static_cast<int>(total_a) != ra || static_cast<int>(total_b) != rb) {
                throw std::invalid_argument(
                    "SectorConstraint: fragment " + std::to_string(f)
                    + " support determinant does not match the fragment's own reference sector");
            }
        }
    }

    if (constraint.n_ct_max < 0) {
        throw std::invalid_argument("SectorConstraint: n_ct_max must be nonnegative");
    }

    for (const auto& det : constraint.protected_determinants) {
        if (!is_las_present(det, constraint)) {
            throw std::invalid_argument(
                "SectorConstraint: a protected determinant is not LAS-present "
                "(P_K must be a subset of P_LAS)");
        }
    }

    constraint.prepared = true;
}

void validate_recovery_policy(const RecoveryPolicy& policy, size_t n_protected,
                              const std::string& context) {
    if (policy.recovery_target == 0) return;
    const size_t required = n_protected + policy.recovery_target;
    if (policy.max_work_dets < required) {
        throw std::invalid_argument(
            context + ": max_work_dets (" + std::to_string(policy.max_work_dets)
            + ") cannot hold P_K (" + std::to_string(n_protected)
            + ") plus the requested recovery_target ("
            + std::to_string(policy.recovery_target) + ")");
    }
}

template<typename StorageType>
std::vector<SectorLabel> classify_initial_pool_t(
    const std::vector<DeterminantT<StorageType>>& initial_pool,
    const SectorConstraint<StorageType>& constraint)
{
    std::vector<SectorLabel> labels;
    labels.reserve(initial_pool.size());
    for (const auto& det : initial_pool) {
        SectorClassification result = classify_determinant_t(det, constraint, /*heat_bath_passed=*/true);
        if (result.label == SectorLabel::LasPresent || result.label == SectorLabel::CtDisallowed) {
            throw std::invalid_argument(
                "SectorConstraint: initial pool contains a nonprotected determinant classified as "
                + std::string(result.label == SectorLabel::LasPresent ? "las_present" : "ct_disallowed")
                + " -- the candidate-generation admission rule would never have produced this "
                  "determinant, so it cannot be seeded directly");
        }
        labels.push_back(result.label);
    }
    return labels;
}

template<typename StorageType>
FilteredGrowthResult compose_and_validate_t(
    const std::vector<DeterminantT<StorageType>>& dets,
    const SectorConstraint<StorageType>& constraint,
    size_t recovery_target,
    const std::string& context)
{
    FilteredGrowthResult out;
    std::unordered_set<DeterminantT<StorageType>> seen_protected;
    for (const auto& det : dets) {
        SectorClassification r = classify_determinant_t(det, constraint, /*heat_bath_passed=*/true);
        switch (r.label) {
            case SectorLabel::Protected:
                out.n_protected++;
                seen_protected.insert(det);
                break;
            case SectorLabel::Dispersion:
            case SectorLabel::CtAllowed:
                out.n_recovery++;
                break;
            case SectorLabel::LasAbsent:
                if (!constraint.allow_las_absent_owned) {
                    throw std::invalid_argument(
                        context + ": determinant set contains a LAS-absent (doorway) determinant "
                        "while allow_las_absent_owned is false");
                }
                out.n_aux++;
                break;
            case SectorLabel::LasPresent:
                throw std::invalid_argument(
                    context + ": determinant set contains a LAS-present determinant outside the "
                    "protected seed -- the admission rule never produces one, so it cannot arrive "
                    "through a warm start, a load, or a phase handoff");
            case SectorLabel::CtDisallowed:
                throw std::invalid_argument(
                    context + ": determinant set contains a charge-transfer determinant of order "
                    + std::to_string(r.n_ct) + " > n_ct_max");
        }
    }
    if (seen_protected.size() != constraint.protected_determinants.size()) {
        throw std::invalid_argument(
            context + ": protected seed identity changed -- " + std::to_string(seen_protected.size())
            + " of " + std::to_string(constraint.protected_determinants.size())
            + " P_K determinants are present");
    }
    out.n_work = dets.size();
    out.recovery_target_met = (recovery_target == 0) || (out.n_recovery >= recovery_target);
    return out;
}

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
        if (fragment_support_lookup_all(det, constraint)) {
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
// Variadic because std::array<uint64_t, N> contains a comma, which the
// preprocessor would otherwise split into extra macro arguments.
#define TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(...)                                          \
    template void validate_and_prepare_sector_constraint<__VA_ARGS__>(                     \
        SectorConstraint<__VA_ARGS__>&, int);                                              \
    template SectorClassification classify_determinant_t<__VA_ARGS__>(                     \
        const DeterminantT<__VA_ARGS__>&, const SectorConstraint<__VA_ARGS__>&, bool);     \
    template std::vector<SectorLabel> classify_initial_pool_t<__VA_ARGS__>(                \
        const std::vector<DeterminantT<__VA_ARGS__>>&, const SectorConstraint<__VA_ARGS__>&); \
    template FilteredGrowthResult compose_and_validate_t<__VA_ARGS__>(                     \
        const std::vector<DeterminantT<__VA_ARGS__>>&, const SectorConstraint<__VA_ARGS__>&, \
        size_t, const std::string&);

TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(uint64_t)
TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(std::array<uint64_t, 2>)
TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(std::array<uint64_t, 3>)
TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(std::array<uint64_t, 4>)
TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(std::array<uint64_t, 5>)
TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(std::array<uint64_t, 6>)
TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(std::array<uint64_t, 7>)
TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT(std::array<uint64_t, 8>)

#undef TRIMCI_INSTANTIATE_SECTOR_CONSTRAINT

} // namespace trimci_core
