#include "screening.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <unordered_set>
#include <atomic>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "../fast_expansion/fe_types.hpp"  // fe_set/fe_map (robin_hood flat hash)
#include "omp_compat.hpp"
#include "bit_compat.hpp"
#include "parallel_sort.hpp"

namespace trimci_core {

// RSS reporting (Linux: reads /proc/self/statm)
static size_t get_rss_mb() {
    std::ifstream statm("/proc/self/statm");
    size_t dummy, rss_pages;
    if (statm >> dummy >> rss_pages) {
        return rss_pages * 4096 / (1024 * 1024);  // pages -> MB
    }
    return 0;
}

// Helper Functions - Precompute double excitation table
// If attentive_orbitals is non-empty, only include (i,j,p,q) where all are in the set
// OPTIMIZED: Parallelized using OpenMP for large systems
DoubleExcTable precompute_double_exc_table(
    int n_orb,
    const std::vector<double>& eri,
    double thr,
    const std::vector<int>& attentive_orbitals
) {
    // Build attentive set for O(1) lookup
    std::unordered_set<int> att_set(attentive_orbitals.begin(), attentive_orbitals.end());
    bool use_attentive = !att_set.empty();
    
    // Determine which orbitals to iterate over
    std::vector<int> orb_list;
    if (use_attentive) {
        orb_list = attentive_orbitals;
        std::sort(orb_list.begin(), orb_list.end());
    } else {
        orb_list.reserve(n_orb);
        for (int o = 0; o < n_orb; ++o) orb_list.push_back(o);
    }
    
    size_t n = orb_list.size();
    
    // Collect all (i,j) pairs for parallel processing
    std::vector<std::pair<int,int>> ij_pairs;
    ij_pairs.reserve(n * (n-1) / 2);
    for (size_t ai = 0; ai < n; ++ai) {
        for (size_t aj = ai + 1; aj < n; ++aj) {
            ij_pairs.push_back({orb_list[ai], orb_list[aj]});
        }
    }
    
    // Parallel compute entries for each (i,j) pair
    std::vector<std::vector<std::tuple<int,int,double>>> results(ij_pairs.size());
    
    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < static_cast<int>(ij_pairs.size()); ++idx) {
        int i = ij_pairs[idx].first;
        int j = ij_pairs[idx].second;
        std::vector<std::tuple<int,int,double>> entries;
        
        for (size_t ap = 0; ap < n; ++ap) {
            int p = orb_list[ap];
            for (size_t aq = ap + 1; aq < n; ++aq) {
                int q = orb_list[aq];
                double h_val = get_eri(eri, n_orb, i, p, j, q) - get_eri(eri, n_orb, i, q, j, p);
                if (std::abs(h_val) > thr) {
                    entries.emplace_back(p, q, h_val);
                }
            }
        }
        
        // Sort by |h_val| descending
        std::sort(entries.begin(), entries.end(),
                  [](auto& a, auto& b){
                      return std::abs(std::get<2>(a)) > std::abs(std::get<2>(b));
                  });
        results[idx] = std::move(entries);
    }
    
    // Merge results into table
    DoubleExcTable table;
    for (size_t idx = 0; idx < ij_pairs.size(); ++idx) {
        if (!results[idx].empty()) {
            table[ij_pairs[idx]] = std::move(results[idx]);
        }
    }
    return table;
}

// Template implementation for processing parent determinants
// If attentive_set is non-empty, all excitations are restricted to those orbitals
template<typename StorageType>
std::vector<std::pair<DeterminantT<StorageType>, double>>
process_parent_worker_t(
    const DeterminantT<StorageType>& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    const std::unordered_set<int>& attentive_set,
    const IntegralSparsityInfo* sparsity
) {
    std::vector<std::pair<DeterminantT<StorageType>, double>> new_pairs;
    auto occ_a = det.getOccupiedAlpha();
    auto occ_b = det.getOccupiedBeta();
    
    bool use_attentive = !attentive_set.empty();

    // αα same-spin double excitations
    for (size_t ia = 0; ia < occ_a.size(); ++ia) {
        for (size_t ib = ia+1; ib < occ_a.size(); ++ib) {
            int i = occ_a[ia], j = occ_a[ib];
            // Skip if i or j not in attentive set
            if (use_attentive && (attentive_set.find(i) == attentive_set.end() || 
                                  attentive_set.find(j) == attentive_set.end())) continue;
            auto it = table.find({i,j});
            if (it == table.end()) continue;
            
            for (const auto& t : it->second) {
                int p, q;
                double h_val;
                std::tie(p, q, h_val) = t;

                // Early break: entries sorted by |h_val| descending
                if (std::abs(h_val) <= thr) break;

                if (BitOps<StorageType>::get_bit(det.alpha, p) ||
                    BitOps<StorageType>::get_bit(det.alpha, q)) continue;

                auto dj = det.doubleExcite(i, j, p, q, true);
                int ph = detail::double_phase_t(det.alpha, i, j, p, q);
                new_pairs.emplace_back(dj, ph * h_val);
            }
        }
    }

    // ββ same-spin double excitations
    for (size_t ia = 0; ia < occ_b.size(); ++ia) {
        for (size_t ib = ia+1; ib < occ_b.size(); ++ib) {
            int i = occ_b[ia], j = occ_b[ib];
            // Skip if i or j not in attentive set
            if (use_attentive && (attentive_set.find(i) == attentive_set.end() || 
                                  attentive_set.find(j) == attentive_set.end())) continue;
            auto it = table.find({i,j});
            if (it == table.end()) continue;
            
            for (const auto& t : it->second) {
                int p, q;
                double h_val;
                std::tie(p, q, h_val) = t;

                // Early break: entries sorted by |h_val| descending
                if (std::abs(h_val) <= thr) break;

                if (BitOps<StorageType>::get_bit(det.beta, p) ||
                    BitOps<StorageType>::get_bit(det.beta, q)) continue;

                auto dj = det.doubleExcite(i, j, p, q, false);
                int ph = detail::double_phase_t(det.beta, i, j, p, q);
                new_pairs.emplace_back(dj, ph * h_val);
            }
        }
    }
    
    // Mixed αβ double excitations
    if (sparsity && !sparsity->ab_exc_table.empty()) {
        // Table path: sorted by |val| desc → early break at threshold
        for (int i : occ_a) {
            if (use_attentive && attentive_set.find(i) == attentive_set.end()) continue;
            for (const auto& entry : sparsity->ab_exc_table[i]) {
                int j = std::get<0>(entry);
                int p = std::get<1>(entry);
                int q = std::get<2>(entry);
                double eri_val = std::get<3>(entry);
                // Early break: entries sorted by |eri_val| descending
                if (std::abs(eri_val) <= thr) break;
                // j must be occupied in beta
                if (!BitOps<StorageType>::get_bit(det.beta, j)) continue;
                // p must be virtual in alpha
                if (BitOps<StorageType>::get_bit(det.alpha, p)) continue;
                // q must be virtual in beta
                if (BitOps<StorageType>::get_bit(det.beta, q)) continue;
                // Attentive checks
                if (use_attentive) {
                    if (attentive_set.find(j) == attentive_set.end()) continue;
                    if (attentive_set.find(p) == attentive_set.end()) continue;
                    if (attentive_set.find(q) == attentive_set.end()) continue;
                }

                StorageType new_alpha = det.alpha;
                StorageType new_beta = det.beta;
                BitOps<StorageType>::clear_bit(new_alpha, i);
                BitOps<StorageType>::set_bit(new_alpha, p);
                BitOps<StorageType>::clear_bit(new_beta, j);
                BitOps<StorageType>::set_bit(new_beta, q);

                DeterminantT<StorageType> dj(new_alpha, new_beta);
                int pa = detail::single_phase_t(det.alpha, i, p);
                int pb = detail::single_phase_t(det.beta, j, q);
                new_pairs.emplace_back(dj, pa * pb * eri_val);
            }
        }
    } else {
        // Dense path (fallback when no ab_exc_table)
        for (int i : occ_a) {
            if (use_attentive && attentive_set.find(i) == attentive_set.end()) continue;
            for (int j : occ_b) {
                if (use_attentive && attentive_set.find(j) == attentive_set.end()) continue;
                auto iterate_p = [&](auto&& callback) {
                    if (use_attentive) {
                        for (int p : attentive_set) callback(p);
                    } else {
                        for (int p = 0; p < n_orb; ++p) callback(p);
                    }
                };
                iterate_p([&](int p) {
                    if (BitOps<StorageType>::get_bit(det.alpha, p)) return;
                    auto iterate_q = [&](auto&& callback_q) {
                        if (use_attentive) {
                            for (int q : attentive_set) callback_q(q);
                        } else {
                            for (int q = 0; q < n_orb; ++q) callback_q(q);
                        }
                    };
                    iterate_q([&](int q) {
                        if (BitOps<StorageType>::get_bit(det.beta, q)) return;
                        double h_val = get_eri(eri, n_orb, i, p, j, q);
                        if (std::abs(h_val) <= thr) return;
                        StorageType new_alpha = det.alpha;
                        StorageType new_beta = det.beta;
                        BitOps<StorageType>::clear_bit(new_alpha, i);
                        BitOps<StorageType>::set_bit(new_alpha, p);
                        BitOps<StorageType>::clear_bit(new_beta, j);
                        BitOps<StorageType>::set_bit(new_beta, q);
                        DeterminantT<StorageType> dj(new_alpha, new_beta);
                        int pa = detail::single_phase_t(det.alpha, i, p);
                        int pb = detail::single_phase_t(det.beta, j, q);
                        new_pairs.emplace_back(dj, pa * pb * h_val);
                    });
                });
            }
        }
    }
    
    // α single excitations
    for (int i : occ_a) {
        if (use_attentive && attentive_set.find(i) == attentive_set.end()) continue;

        if (sparsity && sparsity->eri_is_diagonal) {
            // Sparse path: only h1 neighbors (ERI terms vanish for singles)
            for (int p : sparsity->h1_neighbors[i]) {
                if (use_attentive && attentive_set.find(p) == attentive_set.end()) continue;
                if (BitOps<StorageType>::get_bit(det.alpha, p)) continue;
                auto dj = det.singleExcite(i, p, true);
                double hij = compute_H_ij_t(det, dj, h1, eri, sparsity);
                if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
            }
        } else {
            // Dense path (original)
            auto iterate_p = [&](auto&& callback) {
                if (use_attentive) {
                    for (int p : attentive_set) callback(p);
                } else {
                    for (int p = 0; p < n_orb; ++p) callback(p);
                }
            };
            iterate_p([&](int p) {
                if (BitOps<StorageType>::get_bit(det.alpha, p)) return;
                auto dj = det.singleExcite(i, p, true);
                double hij = compute_H_ij_t(det, dj, h1, eri);
                if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
            });
        }
    }

    // β single excitations
    for (int j : occ_b) {
        if (use_attentive && attentive_set.find(j) == attentive_set.end()) continue;

        if (sparsity && sparsity->eri_is_diagonal) {
            // Sparse path: only h1 neighbors
            for (int q : sparsity->h1_neighbors[j]) {
                if (use_attentive && attentive_set.find(q) == attentive_set.end()) continue;
                if (BitOps<StorageType>::get_bit(det.beta, q)) continue;
                auto dj = det.singleExcite(j, q, false);
                double hij = compute_H_ij_t(det, dj, h1, eri, sparsity);
                if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
            }
        } else {
            // Dense path (original)
            auto iterate_q = [&](auto&& callback) {
                if (use_attentive) {
                    for (int q : attentive_set) callback(q);
                } else {
                    for (int q = 0; q < n_orb; ++q) callback(q);
                }
            };
            iterate_q([&](int q) {
                if (BitOps<StorageType>::get_bit(det.beta, q)) return;
                auto dj = det.singleExcite(j, q, false);
                double hij = compute_H_ij_t(det, dj, h1, eri);
                if (std::abs(hij) > thr) new_pairs.emplace_back(dj, hij);
            });
        }
    }

    return new_pairs;
}

// Wrapper for backward compatibility (process_parent_worker)
std::vector<std::pair<Determinant,double>>
process_parent_worker(
    const Determinant& det,
    int n_orb,
    double thr,
    const DoubleExcTable& table,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri
) {
    // Pass empty attentive_set for backward compatibility (use all orbitals)
    return process_parent_worker_t<uint64_t>(det, n_orb, thr, table, h1, eri, {});
}


// Template implementation for pool building
// Two-stage screening for PT2 modes: heat_bath pre-filter -> PT2 refinement
template<typename StorageType>
std::pair<std::vector<DeterminantT<StorageType>>, double>
pool_build_t(
    const std::vector<DeterminantT<StorageType>>& initial_pool,
    std::vector<double> initial_coeffs,  // P8: by-value for post-loop release
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold,
    size_t target_size,
    HijCacheT<StorageType>& cache,
    const std::string& cache_file,
    const std::vector<int>& attentive_orbitals,
    int verbosity,
    const PoolBuildParams& params,
    const IntegralSparsityInfo* precomputed_sparsity,
    const DoubleExcTable* precomputed_table,
    const SectorConstraint<StorageType>* sector_constraint,
    ScreeningTelemetry* telemetry_out,
    FilteredGrowthResult* filtered_result_out
) {
    // Extract parameters from struct
    const std::string& screening_mode = params.screening_mode;
    double e0 = params.e0;
    int max_rounds = params.max_rounds;
    double threshold_decay = params.threshold_decay;
    double min_threshold = params.min_threshold;
    int max_stagnant_rounds_param = params.max_stagnant_rounds;
    double pt2_denom_min = params.pt2_denom_min;
    
    // Normalize screening_mode aliases: "hb" → "heat_bath", "hb-pt2" → "heat_bath_pt2"
    std::string mode = screening_mode;
    if (mode == "hb") mode = "heat_bath";
    else if (mode == "hb-pt2") mode = "heat_bath_pt2";

    // Determine effective strategy factor
    int effective_factor;
    if (params.strategy_factor > 0) {
        effective_factor = params.strategy_factor;
    } else if (mode == "uniform" || mode == "heat_bath") {
        effective_factor = 1;  // heat_bath/uniform: no overgeneration, fill-and-stop
    } else {
        effective_factor = 20;  // 20× overgeneration for PT2/MI score-based modes
    }
    size_t effective_target = target_size * effective_factor;

    // Determine screening mode flags
    bool use_pt2_denominator = (mode == "heat_bath_pt2" || mode == "pt2");
    bool use_pt2_aggregation = (mode == "pt2");
    bool use_mi_weights = (mode == "mi") && !params.mi_weights.empty();
    const double mi_alpha = params.mi_alpha;
    const int mi_n = params.mi_n_orb;
    
    if (verbosity >= 1) {
        std::cout << "[PoolBuild] Starting pool build: "
              << "target_size=" << target_size
              << ", threshold=" << threshold
              << ", mode=" << screening_mode
              << ", factor=" << effective_factor;
        if (use_pt2_denominator) {
            std::cout << ", e0=" << e0;
        }
        if (!attentive_orbitals.empty()) {
            std::cout << ", attentive_orbitals=" << attentive_orbitals.size();
        }
        std::cout << std::endl;
    }

    // Build attentive set for O(1) lookup
    std::unordered_set<int> attentive_set(attentive_orbitals.begin(), attentive_orbitals.end());

    // Use precomputed table if available, otherwise build locally
    DoubleExcTable local_table;
    const DoubleExcTable* table_ptr;
    if (precomputed_table) {
        table_ptr = precomputed_table;
        if (verbosity >= 2) {
            std::cout << "[PoolBuild] Using precomputed double_exc_table, entries=" << precomputed_table->size() << std::endl;
        }
    } else {
        auto precompute_start = std::chrono::high_resolution_clock::now();
        local_table = precompute_double_exc_table(n_orb, eri, min_threshold, attentive_orbitals);
        auto precompute_end = std::chrono::high_resolution_clock::now();
        double precompute_time = std::chrono::duration<double>(precompute_end - precompute_start).count();
        if (verbosity >= 2) {
            std::cout << "[PoolBuild] precompute_double_exc_table: " << precompute_time << "s, entries=" << local_table.size() << std::endl;
        }
        table_ptr = &local_table;
    }

    // Use precomputed sparsity info if available, otherwise build
    // Always pass sparsity — ab_exc_table enables αβ early break even for dense systems
    IntegralSparsityInfo local_sparsity_info;
    const IntegralSparsityInfo* sparsity_ptr = precomputed_sparsity;
    if (!sparsity_ptr) {
        local_sparsity_info = build_sparsity_info(n_orb, h1, eri);
        sparsity_ptr = &local_sparsity_info;
    }
    if (verbosity >= 1 && sparsity_ptr->is_sparse) {
        std::cout << "[PoolBuild] Sparsity detected: h1=" << sparsity_ptr->h1_sparsity
                  << ", eri=" << sparsity_ptr->eri_sparsity
                  << ", diagonal=" << sparsity_ptr->eri_is_diagonal << std::endl;
    }

    if (verbosity >= 2) std::cout << "[PoolBuild] RSS before hash tables: " << get_rss_mb() << " MB" << std::endl;

    // P5: single pool_map replaces pool_set + score_map.
    // Value = screening score; initial dets get INFINITY to guarantee retention in strict truncation.
    fe::fe_map<DeterminantT<StorageType>, double, fe::DetHash<StorageType>> pool_map;
    pool_map.reserve(target_size);
    for (auto& d : initial_pool) {
        pool_map[d] = std::numeric_limits<double>::infinity();
    }
    if (verbosity >= 2) std::cout << "[PoolBuild] RSS after pool_map(reserve+fill " << pool_map.size() << "): " << get_rss_mb() << " MB" << std::endl;

    // Milestone C2a: accumulates across every round/chunk of the loop below.
    // Untouched when sector_constraint is nullptr (disabled path).
    ScreeningTelemetry aggregate_telemetry;

    // Milestone C2b-1: filtered-growth policy is an explicit opt-in
    // (sector_constraint->recovery_target > 0). A constraint can be active
    // purely for candidate filtering (C2a) while pool_build_t still uses
    // classic pool_map.size() stopping -- this flag, not sector_constraint
    // alone, is what switches the stopping/truncation/accounting behavior.
    const bool filtered_growth_active = sector_constraint && sector_constraint->recovery_target > 0;
    FilteredGrowthResult filtered_result;
    fe::fe_map<DeterminantT<StorageType>, SectorLabel, fe::DetHash<StorageType>> pool_labels;
    if (filtered_growth_active) {
        // Classify the initial pool ONCE. Throws if a nonprotected member is
        // las_present or ct_disallowed -- the admission rule would never have
        // produced it, so it cannot be seeded directly.
        std::vector<SectorLabel> initial_labels = classify_initial_pool_t(initial_pool, *sector_constraint);
        for (size_t i = 0; i < initial_pool.size(); ++i) {
            pool_labels[initial_pool[i]] = initial_labels[i];
            switch (initial_labels[i]) {
                case SectorLabel::Protected: filtered_result.n_protected++; break;
                case SectorLabel::Dispersion:
                case SectorLabel::CtAllowed: filtered_result.n_recovery++; break;
                case SectorLabel::LasAbsent: filtered_result.n_aux++; break;
                default: break;
            }
        }
        filtered_result.n_work = pool_map.size();
    }
    // Unifies the two stopping regimes: classic (pool_map.size() vs
    // target_size, C2a and earlier, unchanged) and filtered growth (unique
    // N_recovery vs recovery_target, C2b-1, never total pool size).
    auto growth_metric = [&]() -> size_t {
        return filtered_growth_active ? static_cast<size_t>(filtered_result.n_recovery) : pool_map.size();
    };
    const size_t growth_target = filtered_growth_active ? sector_constraint->recovery_target : target_size;

    // Shared by all three insertion sites (heat_bath merge, PT2 aggregation,
    // PT2 heat_bath_pt2) so the doorway/work-memory/composition rules are
    // defined exactly once, not per candidate-buffer shape.
    auto filtered_insertion_blocked = [&](SectorLabel label) -> bool {
        if (!filtered_growth_active) return false;
        return label == SectorLabel::LasAbsent && filtered_result.n_aux >= sector_constraint->doorway_limit;
    };
    auto filtered_work_memory_full = [&]() -> bool {
        return filtered_growth_active && filtered_result.n_work >= sector_constraint->max_work_memory;
    };
    auto account_filtered_insertion = [&](const DeterminantT<StorageType>& det, SectorLabel label) {
        pool_labels[det] = label;
        switch (label) {
            case SectorLabel::Protected: filtered_result.n_protected++; break;
            case SectorLabel::Dispersion:
            case SectorLabel::CtAllowed: filtered_result.n_recovery++; break;
            case SectorLabel::LasAbsent:
                filtered_result.n_aux++;
                if (filtered_result.n_aux >= sector_constraint->doorway_limit) {
                    filtered_result.doorway_limit_reached = true;
                }
                break;
            default: break;  // LasPresent/CtDisallowed never reach an insertion site
        }
        filtered_result.n_work++;
    };

    // Control whether to use coefficient map
    bool use_coeffs = !initial_coeffs.empty();

    // Compute max |H_ij| upper bound for frontier pre-filtering.
    // Dets with |c_i| < threshold / max_hij can never produce candidates
    // (their local_threshold = threshold / |c_i| > max_hij), so skip them.
    double max_hij = 0.0;
    if (use_coeffs && !initial_pool.empty()) {
        double max_h1_val = 0.0, max_eri_val = 0.0;
        for (int i = 0; i < n_orb; ++i)
            for (int j = 0; j < n_orb; ++j)
                max_h1_val = std::max(max_h1_val, std::abs(h1[i][j]));
        for (size_t k = 0; k < eri.size(); ++k)
            max_eri_val = std::max(max_eri_val, std::abs(eri[k]));
        int n_elec = static_cast<int>(initial_pool[0].getOccupiedAlpha().size()
                                    + initial_pool[0].getOccupiedBeta().size());
        max_hij = max_h1_val + n_elec * max_eri_val;
    }

    // P6: frontier_coeffs replaces coeff_map in single_pass_mode (max_rounds==1).
    // Returns (dets, coeffs) pair; coeffs aligned with dets for O(1) lookup by index.
    bool single_pass_mode = (max_rounds == 1);
    std::vector<double> frontier_coeffs;

    auto build_filtered_frontier = [&](double thr) {
        std::vector<DeterminantT<StorageType>> filt;
        frontier_coeffs.clear();
        if (!use_coeffs || max_hij <= 0.0) {
            filt.assign(initial_pool.begin(), initial_pool.end());
            if (use_coeffs) frontier_coeffs = initial_coeffs;
            return filt;
        }
        double ci_min = thr / max_hij;

        // Argsort: collect indices of dets passing filter, sort by |c_i| desc.
        std::vector<size_t> idx;
        idx.reserve(initial_pool.size());
        for (size_t i = 0; i < initial_pool.size(); ++i) {
            if (std::abs(initial_coeffs[i]) >= ci_min)
                idx.push_back(i);
        }
        trimci::parallel_sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) {
                      return std::abs(initial_coeffs[a]) > std::abs(initial_coeffs[b]);
                  });
        filt.reserve(idx.size());
        frontier_coeffs.reserve(idx.size());
        for (size_t i : idx) {
            filt.push_back(initial_pool[i]);
            frontier_coeffs.push_back(initial_coeffs[i]);
        }

        if (verbosity >= 1) {
            std::cout << "[PoolBuild] Frontier filtered: " << initial_pool.size()
                      << " -> " << filt.size()
                      << " (|c_i| >= " << std::scientific << std::setprecision(2)
                      << ci_min << std::fixed
                      << "), sorted by |c_i| desc" << std::endl;
        }
        return filt;
    };

    std::vector<DeterminantT<StorageType>> frontier = build_filtered_frontier(threshold);
    if (verbosity >= 2) {
        std::cout << "[PoolBuild] use_coeffs: " << use_coeffs << ", screening_mode: " << screening_mode << std::endl;
    }

    // P6: coeff_map only needed in multi-round mode (frontier changes across rounds).
    // In single_pass_mode, frontier_coeffs provides O(1) lookup by frontier index.
    fe::fe_map<DeterminantT<StorageType>, double, fe::DetHash<StorageType>> coeff_map;
    if (use_coeffs && !single_pass_mode) {
        coeff_map.reserve(target_size);
        if (verbosity >= 2) std::cout << "[PoolBuild] RSS after coeff_map reserve: " << get_rss_mb() << " MB" << std::endl;
        for (size_t i = 0; i < initial_pool.size(); ++i) {
            coeff_map[initial_pool[i]] = initial_coeffs[i];
        }
        if (verbosity >= 2) std::cout << "[PoolBuild] RSS after coeff_map fill: " << get_rss_mb() << " MB" << std::endl;
    } else if (use_coeffs) {
        if (verbosity >= 2) std::cout << "[PoolBuild] P6: using frontier_coeffs (no coeff_map)" << std::endl;
    }

    int round = 1;
    std::atomic<bool> reached{false};
    size_t prev_pool_size = pool_map.size();
    int stagnant_rounds = 0;
    // Milestone C2b-1: distinct from `reached` (which means "target met") --
    // this means "growth cannot continue and the target was NOT met," e.g. the
    // independent working-memory ceiling. Latches permanently once set.
    bool filtered_saturated = false;

    while (growth_metric() < growth_target) {
        if (frontier.empty() || (max_rounds > 0 && round > max_rounds)) {
            bool round_limit_hit = (max_rounds > 0 && round > max_rounds) && !frontier.empty();
            if (growth_metric() < growth_target) {
                // Check if threshold is too small or no progress is being made
                if (threshold < min_threshold) {
                if (verbosity >= 1) {
                    std::cout << "[PoolBuild] threshold too small (" << std::scientific << std::setprecision(4) << threshold
                              << " < " << min_threshold << std::defaultfloat << "), stopping to prevent infinite loop." << std::endl;
                }
                    if (filtered_growth_active) filtered_result.saturation_reason = SaturationReason::ThresholdFloor;
                    break;
                }
                size_t current_pool = growth_metric();
                size_t gained = current_pool - prev_pool_size;
                if (gained == 0) {
                    stagnant_rounds++;
                    if (stagnant_rounds >= max_stagnant_rounds_param) {
                        if (verbosity >= 1) {
                            std::cout << "[PoolBuild] no progress for " << max_stagnant_rounds_param
                                  << " threshold reductions, stopping." << std::endl;
                        }
                        if (filtered_growth_active) {
                            filtered_result.saturation_reason = round_limit_hit
                                ? SaturationReason::RoundLimit : SaturationReason::FrontierExhausted;
                        }
                        break;
                    }
                } else {
                    stagnant_rounds = 0;
                }
                prev_pool_size = current_pool;

                // Relax threshold and restart
                double old_threshold = threshold;
                threshold *= threshold_decay;
                round = 1;
                frontier = build_filtered_frontier(threshold);
                if (verbosity >= 1) {
                    std::cout << "[PoolBuild] pool=" << current_pool
                          << "/" << growth_target
                          << " (+" << gained << " new), threshold "
                          << std::scientific << std::setprecision(4)
                          << old_threshold << " → " << threshold
                          << std::defaultfloat
                          << ", frontier=" << frontier.size()
                          << ", stagnant=" << stagnant_rounds
                          << "/" << max_stagnant_rounds_param
                          << std::endl;
                }
            } else {
                break;
            }
        }

        if (verbosity >= 2) {
            std::cout << "[PoolBuild] Round " << round
                  << ": pool_size=" << pool_map.size()
                  << ", frontier_size=" << frontier.size()
                  << std::endl;
        }

        std::vector<DeterminantT<StorageType>> new_frontier;
        new_frontier.reserve(filtered_growth_active
            ? frontier.size()
            : (target_size > pool_map.size() ? target_size - pool_map.size() : size_t(0)));

        // Candidate definition for deferred processing (PT2 modes only)
        // Added h_jj for PT2 denominator, screening_score for final ranking
        struct Candidate {
            DeterminantT<StorageType> parent;
            DeterminantT<StorageType> child;
            double hij;
            double est_cj;
            double h_jj;            // Diagonal element H_jj (for PT2)
            double screening_score; // Final score used for selection
            SectorLabel label;      // Milestone C2b-1: unused unless sector_constraint is set
        };

        // Lightweight candidate for heat_bath mode (32 bytes vs 64 bytes)
        // Drops: parent det, hij, h_jj — not needed for heat_bath merge
        struct LightCandidate {
            DeterminantT<StorageType> child;
            double score;      // |hij * ci|
            double parent_ci;  // parent's |c_i| for coeff_map in subsequent rounds
            SectorLabel label; // Milestone C2b-1: unused unless sector_constraint is set
        };

        #ifdef _OPENMP
        int n_threads = omp_get_max_threads();
        #else
        int n_threads = 1;
        #endif

        // ============================================================
        // Bifurcate: heat_bath uses chunked streaming, PT2 uses full pass
        // ============================================================
        if (!use_pt2_denominator) {
            // ---- CHUNKED heat_bath mode ----
            // Process frontier in chunks to bound peak candidate buffer memory.
            // Each chunk: OMP screen → merge into pool_set → free buffers.
            const size_t CHUNK_SIZE = 5'000'000;  // 5M dets per chunk
            const size_t total_chunks = (frontier.size() + CHUNK_SIZE - 1) / CHUNK_SIZE;
            size_t chunks_processed = 0;
            size_t pool_before_chunks = pool_map.size();
            std::vector<std::vector<LightCandidate>> thread_light_candidates(n_threads);
            std::vector<ScreeningTelemetry> thread_telemetry(sector_constraint ? n_threads : 0);
            size_t total_candidates_processed = 0;
            size_t consecutive_zero_chunks = 0;
            const size_t MAX_ZERO_CHUNKS = 10;

            for (size_t chunk_start = 0; chunk_start < frontier.size(); chunk_start += CHUNK_SIZE) {
                size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, frontier.size());
                ++chunks_processed;

                // Clear buffers (capacity retained for reuse)
                for (auto& v : thread_light_candidates) v.clear();
                for (auto& t : thread_telemetry) t.reset();

                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    #pragma omp for schedule(dynamic)
                    for (int idx = chunk_start; idx < static_cast<int>(chunk_end); ++idx) {
                        auto det = frontier[idx];
                        double ci = 1.0;
                        if (use_coeffs) {
                            // P6: frontier_coeffs for single_pass, coeff_map for multi-round
                            if (single_pass_mode) {
                                ci = frontier_coeffs[idx];
                            } else {
                                auto it = coeff_map.find(det);
                                ci = (it != coeff_map.end()) ? it->second : 1.0;
                            }
                        }
                        double abs_ci = std::abs(ci);
                        double local_threshold = threshold / std::max(abs_ci, 1e-12);

                        auto locals = process_parent_worker_t(det, n_orb, local_threshold, *table_ptr, h1, eri, attentive_set, sparsity_ptr);
                        if (sector_constraint) thread_telemetry[tid].generated += locals.size();

                        for (auto& pr : locals) {
                            const auto& dj = pr.first;
                            double hij = pr.second;
                            bool hb_pass = std::abs(hij) > local_threshold;
                            bool sector_admit = true;
                            SectorLabel candidate_label = SectorLabel::Dispersion;  // unused unless sector_constraint
                            if (hb_pass && sector_constraint) {
                                SectorClassification result = screen_candidate_t(dj, *sector_constraint);
                                candidate_label = result.label;
                                int label_idx = static_cast<int>(result.label);
                                thread_telemetry[tid].hb_passing[label_idx]++;
                                if (result.admitted) {
                                    thread_telemetry[tid].admitted[label_idx]++;
                                } else {
                                    thread_telemetry[tid].rejected[label_idx]++;
                                }
                                sector_admit = result.admitted;
                            }
                            if (hb_pass && sector_admit && pool_map.count(dj) == 0) {
                                double heat_bath_score = std::abs(hij * abs_ci);
                                // MI-weighted scoring: boost excitations involving correlated orbitals
                                if (use_mi_weights) {
                                    // Find changed orbitals by comparing occupations
                                    auto occ_a_p = det.getOccupiedAlpha();
                                    auto occ_a_c = dj.getOccupiedAlpha();
                                    auto occ_b_p = det.getOccupiedBeta();
                                    auto occ_b_c = dj.getOccupiedBeta();
                                    // Collect hole/particle orbitals
                                    int changed[8]; int nc = 0;
                                    for (int o : occ_a_p)
                                        if (std::find(occ_a_c.begin(), occ_a_c.end(), o) == occ_a_c.end())
                                            if (nc < 8) changed[nc++] = o;  // hole
                                    for (int o : occ_a_c)
                                        if (std::find(occ_a_p.begin(), occ_a_p.end(), o) == occ_a_p.end())
                                            if (nc < 8) changed[nc++] = o;  // particle
                                    for (int o : occ_b_p)
                                        if (std::find(occ_b_c.begin(), occ_b_c.end(), o) == occ_b_c.end())
                                            if (nc < 8) changed[nc++] = o;
                                    for (int o : occ_b_c)
                                        if (std::find(occ_b_p.begin(), occ_b_p.end(), o) == occ_b_p.end())
                                            if (nc < 8) changed[nc++] = o;
                                    // Max MI weight among all pairs of changed orbitals
                                    double mi_w = 0.0;
                                    for (int ii = 0; ii < nc; ++ii)
                                        for (int jj = ii+1; jj < nc; ++jj) {
                                            double w = params.mi_weights[changed[ii] * mi_n + changed[jj]];
                                            if (w > mi_w) mi_w = w;
                                        }
                                    heat_bath_score *= (1.0 + mi_alpha * mi_w);
                                }
                                thread_light_candidates[tid].push_back({dj, heat_bath_score, abs_ci, candidate_label});
                            }
                        }
                    }
                } // omp parallel end

                if (sector_constraint) {
                    for (const auto& t : thread_telemetry) aggregate_telemetry.merge(t);
                }

                // P5: Merge this chunk's candidates into pool_map (MAX score update)
                size_t chunk_candidates = 0;
                bool pool_capped = false;
                bool work_memory_capped = false;
                for (int t = 0; t < n_threads; ++t) {
                    for (const auto& cand : thread_light_candidates[t]) {
                        // max_pool_size: stop merging when pool is large enough
                        // Candidates from earlier threads (higher-weight parents) are kept;
                        // later candidates (lower-weight parents) are discarded.
                        if (params.max_pool_size > 0 && pool_map.size() >= static_cast<size_t>(params.max_pool_size)) {
                            pool_capped = true;
                            break;
                        }
                        // Milestone C2b-1: the doorway limit blocks ONLY new las_absent (O)
                        // insertions -- R and P_K must still be able to enter after it is
                        // reached, so this is a `continue` (skip this one candidate), never
                        // a `break` (stop the whole merge).
                        if (filtered_insertion_blocked(cand.label)) {
                            continue;
                        }
                        // The independent working-memory ceiling, unlike the doorway limit,
                        // blocks every class once N_work is exhausted.
                        if (filtered_work_memory_full()) {
                            work_memory_capped = true;
                            break;
                        }
                        auto [it, inserted] = pool_map.try_emplace(cand.child, cand.score);
                        if (inserted) {
                            new_frontier.push_back(cand.child);
                            if (use_coeffs && !single_pass_mode) {
                                coeff_map[cand.child] = cand.parent_ci;
                            }
                            if (filtered_growth_active) account_filtered_insertion(cand.child, cand.label);
                        } else if (use_coeffs && cand.score > it->second && !std::isinf(it->second)) {
                            // Update to MAX score (skip initial dets with INFINITY)
                            it->second = cand.score;
                        }
                    }
                    chunk_candidates += thread_light_candidates[t].size();
                    if (pool_capped || work_memory_capped) break;
                }
                total_candidates_processed += chunk_candidates;

                if (pool_capped) {
                    if (verbosity >= 1) {
                        std::cout << "[PoolBuild] max_pool_size reached (" << params.max_pool_size
                                  << "), stopping merge." << std::endl;
                    }
                    reached.store(true);
                    break;
                }
                if (work_memory_capped) {
                    if (verbosity >= 1) {
                        std::cout << "[PoolBuild] max_work_memory reached ("
                                  << sector_constraint->max_work_memory << "), stopping merge." << std::endl;
                    }
                    filtered_result.saturation_reason = SaturationReason::WorkingMemoryCeiling;
                    filtered_saturated = true;
                    break;
                }

                if (verbosity >= 2) {
                    std::cout << "[PoolBuild]   chunk " << chunks_processed
                              << "/" << total_chunks
                              << ": pool=" << pool_map.size()
                              << " (+" << (pool_map.size() - pool_before_chunks) << " new)"
                              << ", chunk_cands=" << chunk_candidates
                              << ", RSS=" << get_rss_mb() << " MB"
                              << std::endl;
                }

                if (params.early_stop && growth_metric() >= growth_target) {
                    reached.store(true);
                    break;
                }

                // Early exit if frontier exhausted (consecutive zero-yield chunks)
                if (chunk_candidates == 0) {
                    if (++consecutive_zero_chunks >= MAX_ZERO_CHUNKS) {
                        if (verbosity >= 1) {
                            std::cout << "[PoolBuild] " << MAX_ZERO_CHUNKS
                                      << " consecutive zero-yield chunks, breaking early at chunk "
                                      << chunks_processed << "/" << total_chunks << std::endl;
                        }
                        break;
                    }
                } else {
                    consecutive_zero_chunks = 0;
                }
            }

            // Free candidate buffers
            { std::vector<std::vector<LightCandidate>>().swap(thread_light_candidates); }

            if (verbosity >= 2) {
                std::cout << "[PoolBuild] Streamed " << total_candidates_processed
                          << " candidates in "
                          << chunks_processed << "/" << total_chunks
                          << " chunks, pool=" << pool_map.size()
                          << " (+" << (pool_map.size() - pool_before_chunks) << " new)"
                          << std::endl;
            }

            // Check target after all chunks
            if (growth_metric() >= growth_target) {
                reached.store(true);
            }
        } else {
        // ---- PT2 modes: full pass (need all candidates for sorting) ----
        std::vector<std::vector<Candidate>> thread_candidates(n_threads);
        std::vector<ScreeningTelemetry> thread_telemetry_pt2(sector_constraint ? n_threads : 0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(dynamic)
            for (int idx = 0; idx < static_cast<int>(frontier.size()); ++idx) {
                auto det = frontier[idx];
                double ci = 1.0;
                if (use_coeffs) {
                    if (single_pass_mode) {
                        ci = frontier_coeffs[idx];
                    } else {
                        auto it = coeff_map.find(det);
                        ci = (it != coeff_map.end()) ? it->second : 1.0;
                    }
                }
                double abs_ci = std::abs(ci);
                double local_threshold = threshold / std::max(abs_ci, 1e-12);

                auto locals = process_parent_worker_t(det, n_orb, local_threshold, *table_ptr, h1, eri, attentive_set, sparsity_ptr);
                if (sector_constraint) thread_telemetry_pt2[tid].generated += locals.size();

                for (auto& pr : locals) {
                    const auto& dj = pr.first;
                    double hij = pr.second;
                    bool hb_pass = std::abs(hij) > local_threshold;
                    bool sector_admit = true;
                    SectorLabel candidate_label = SectorLabel::Dispersion;  // unused unless sector_constraint
                    if (hb_pass && sector_constraint) {
                        SectorClassification result = screen_candidate_t(dj, *sector_constraint);
                        candidate_label = result.label;
                        int label_idx = static_cast<int>(result.label);
                        thread_telemetry_pt2[tid].hb_passing[label_idx]++;
                        if (result.admitted) {
                            thread_telemetry_pt2[tid].admitted[label_idx]++;
                        } else {
                            thread_telemetry_pt2[tid].rejected[label_idx]++;
                        }
                        sector_admit = result.admitted;
                    }
                    if (hb_pass && sector_admit) {
                        double heat_bath_score = std::abs(hij * abs_ci);
                        thread_candidates[tid].push_back({det, dj, hij, abs_ci, 0.0, heat_bath_score, candidate_label});
                    }
                }
            }
        } // omp parallel end

        if (sector_constraint) {
            for (const auto& t : thread_telemetry_pt2) aggregate_telemetry.merge(t);
        }

        if (verbosity >= 2) {
            size_t total_heavy = 0;
            for (int t = 0; t < n_threads; ++t)
                total_heavy += thread_candidates[t].size();
            std::cout << "[PoolBuild] PT2 candidate buffer: "
                      << total_heavy << " (" << (total_heavy * sizeof(Candidate) >> 20) << " MB)"
                      << std::endl;
        }

        // Stage 2: PT2 merge and refine
            // PT2 modes: Two-stage screening
            // First collect all candidates
            std::vector<Candidate> all_candidates;
            for (int t = 0; t < n_threads; ++t) {
                all_candidates.insert(all_candidates.end(), 
                                     thread_candidates[t].begin(), 
                                     thread_candidates[t].end());
            }
            
            if (verbosity >= 2) {
                std::cout << "[PoolBuild] Stage 1 collected " << all_candidates.size() 
                          << " candidates (need " << (target_size - pool_map.size()) << ")" << std::endl;
            }
            
            // Sort by heat_bath score for pre-filtering
            std::sort(all_candidates.begin(), all_candidates.end(),
                     [](const Candidate& a, const Candidate& b) {
                         return a.screening_score > b.screening_score;
                     });
            
            // Take top effective_target candidates by heat_bath score
            size_t pre_filter_count = std::min(all_candidates.size(), effective_target);
            
            if (verbosity >= 2) {
                std::cout << "[PoolBuild] Stage 2: computing PT2 for top " << pre_filter_count 
                          << " candidates" << std::endl;
            }
            
            // Compute H_jj and PT2 score for pre-filtered candidates
            for (size_t i = 0; i < pre_filter_count; ++i) {
                auto& cand = all_candidates[i];
                // Compute H_jj (diagonal element of child)
                cand.h_jj = compute_H_ij_t(cand.child, cand.child, h1, eri);
                double denom = e0 - cand.h_jj;
                
                // Avoid division by zero for intruder states
                if (std::abs(denom) < pt2_denom_min) {
                    denom = (denom >= 0) ? pt2_denom_min : -pt2_denom_min;
                }
                
                // PT2 score = |H_ij * c_i|^2 / |E_0 - H_jj|
                double hij_ci = cand.hij * cand.est_cj;
                cand.screening_score = std::abs((hij_ci * hij_ci) / denom);
            }
            
            // Re-sort pre-filtered candidates by PT2 score
            auto pt2_candidates_begin = all_candidates.begin();
            auto pt2_candidates_end = all_candidates.begin() + pre_filter_count;
            std::sort(pt2_candidates_begin, pt2_candidates_end,
                     [](const Candidate& a, const Candidate& b) {
                         return a.screening_score > b.screening_score;
                     });
            
            // For full PT2 mode: aggregate contributions from multiple parents
            if (use_pt2_aggregation) {
                fe::fe_map<DeterminantT<StorageType>, double, fe::DetHash<StorageType>> child_aggregated_score;
                fe::fe_map<DeterminantT<StorageType>, DeterminantT<StorageType>, fe::DetHash<StorageType>> child_best_parent;
                fe::fe_map<DeterminantT<StorageType>, double, fe::DetHash<StorageType>> child_best_hij;
                fe::fe_map<DeterminantT<StorageType>, double, fe::DetHash<StorageType>> child_best_score;
                fe::fe_map<DeterminantT<StorageType>, double, fe::DetHash<StorageType>> child_best_est_cj;
                fe::fe_map<DeterminantT<StorageType>, SectorLabel, fe::DetHash<StorageType>> child_label;

                for (size_t i = 0; i < pre_filter_count; ++i) {
                    const auto& cand = all_candidates[i];
                    auto it = child_aggregated_score.find(cand.child);
                    if (it == child_aggregated_score.end()) {
                        child_aggregated_score[cand.child] = cand.screening_score;
                        child_best_parent[cand.child] = cand.parent;
                        child_best_hij[cand.child] = cand.hij;
                        child_best_score[cand.child] = cand.screening_score;
                        child_best_est_cj[cand.child] = cand.est_cj;
                        child_label[cand.child] = cand.label;
                    } else {
                        it->second += cand.screening_score;
                        if (cand.screening_score > child_best_score[cand.child]) {
                            child_best_score[cand.child] = cand.screening_score;
                            child_best_parent[cand.child] = cand.parent;
                            child_best_hij[cand.child] = cand.hij;
                            child_best_est_cj[cand.child] = cand.est_cj;
                        }
                    }
                }

                // Sort by aggregated score
                std::vector<std::pair<DeterminantT<StorageType>, double>> sorted_children;
                sorted_children.reserve(child_aggregated_score.size());
                for (const auto& kv : child_aggregated_score) {
                    sorted_children.emplace_back(kv.first, kv.second);
                }
                std::sort(sorted_children.begin(), sorted_children.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });

                for (const auto& [child, agg_score] : sorted_children) {
                    SectorLabel label = filtered_growth_active ? child_label[child] : SectorLabel::Dispersion;
                    if (filtered_insertion_blocked(label)) continue;
                    if (filtered_work_memory_full()) { filtered_saturated = true;
                        filtered_result.saturation_reason = SaturationReason::WorkingMemoryCeiling; break; }
                    auto [it, inserted] = pool_map.try_emplace(child, agg_score);
                    if (inserted) {
                        new_frontier.push_back(child);
                        if (use_coeffs && !single_pass_mode) {
                            coeff_map[child] = child_best_est_cj[child];
                        }
                        if (filtered_growth_active) account_filtered_insertion(child, label);
                    }
                }
                // Check target after processing all aggregated candidates
                if (growth_metric() >= growth_target) {
                    reached.store(true);
                }
            } else {
                // heat_bath_pt2: add pre-filtered candidates in PT2 score order
                for (size_t i = 0; i < pre_filter_count; ++i) {
                    const auto& cand = all_candidates[i];
                    if (filtered_insertion_blocked(cand.label)) continue;
                    if (filtered_work_memory_full()) { filtered_saturated = true;
                        filtered_result.saturation_reason = SaturationReason::WorkingMemoryCeiling; break; }
                    auto [it, inserted] = pool_map.try_emplace(cand.child, cand.screening_score);
                    if (inserted) {
                        new_frontier.push_back(cand.child);
                        if (use_coeffs && !single_pass_mode) {
                            coeff_map[cand.child] = cand.est_cj;
                        }
                        if (filtered_growth_active) account_filtered_insertion(cand.child, cand.label);
                    }
                }
                // Check target after processing all pre-filtered candidates
                if (growth_metric() >= growth_target) {
                    reached.store(true);
                }
            }
        } // end PT2 modes

        if (reached.load() || filtered_saturated) {
            if (verbosity >= 1) {
                std::cout << (reached.load() ? "[PoolBuild] target_size reached, stopping.\n"
                                             : "[PoolBuild] filtered growth saturated, stopping.\n");
            }
            frontier.swap(new_frontier);
            break;
        }

        frontier.swap(new_frontier);
        ++round;
    }

    if (verbosity >= 1) {
        std::cout << "[PoolBuild] Final pool size: " << pool_map.size()
              << "/" << target_size
              << ", final threshold: " << std::scientific << std::setprecision(4) << threshold
              << std::defaultfloat << std::endl;
    }

    // P8: release initial_coeffs (by-value copy, no longer needed)
    { std::vector<double> tmp; tmp.swap(initial_coeffs); }
    { std::vector<double> tmp; tmp.swap(frontier_coeffs); }

    // Extract keys from pool_map
    std::vector<DeterminantT<StorageType>> final_pool;
    final_pool.reserve(pool_map.size());
    for (const auto& [det, score] : pool_map) {
        final_pool.push_back(det);
    }
    
    // Milestone C2b-1: category-aware strict truncation for filtered growth --
    // a DIFFERENT contract from the classic branch below, not an extension of
    // it. Always retains every P_K member (not "the whole initial pool" --
    // C2b-1 explicitly does not auto-protect nonprotected initial-pool
    // members), retains at most recovery_target R members by score, and at
    // most doorway_limit O members by score.
    if (params.strict_target_size && filtered_growth_active) {
        auto score_of = [&](const DeterminantT<StorageType>& d) -> double {
            auto it = pool_map.find(d);
            return (it != pool_map.end()) ? std::abs(it->second) : 0.0;
        };
        std::vector<DeterminantT<StorageType>> kept_protected, kept_recovery, kept_aux;
        for (auto& d : final_pool) {
            auto it = pool_labels.find(d);
            SectorLabel label = (it != pool_labels.end()) ? it->second : SectorLabel::Dispersion;
            switch (label) {
                case SectorLabel::Protected: kept_protected.push_back(d); break;
                case SectorLabel::Dispersion:
                case SectorLabel::CtAllowed: kept_recovery.push_back(d); break;
                case SectorLabel::LasAbsent: kept_aux.push_back(d); break;
                default: break;  // unreachable: never inserted with this label
            }
        }
        auto by_score_desc = [&](const DeterminantT<StorageType>& a, const DeterminantT<StorageType>& b) {
            return score_of(a) > score_of(b);
        };
        std::sort(kept_recovery.begin(), kept_recovery.end(), by_score_desc);
        if (kept_recovery.size() > sector_constraint->recovery_target) {
            kept_recovery.resize(sector_constraint->recovery_target);
        }
        std::sort(kept_aux.begin(), kept_aux.end(), by_score_desc);
        if (sector_constraint->doorway_limit != std::numeric_limits<size_t>::max()
            && kept_aux.size() > sector_constraint->doorway_limit) {
            kept_aux.resize(sector_constraint->doorway_limit);
        }
        if (verbosity >= 1) {
            std::cout << "[PoolBuild] Filtered strict truncation: P_K=" << kept_protected.size()
                      << ", R " << final_pool.size() << "->" << kept_recovery.size()
                      << ", O " << kept_aux.size() << std::endl;
        }
        final_pool.clear();
        final_pool.insert(final_pool.end(), kept_protected.begin(), kept_protected.end());
        final_pool.insert(final_pool.end(), kept_recovery.begin(), kept_recovery.end());
        final_pool.insert(final_pool.end(), kept_aux.begin(), kept_aux.end());
    } else if (params.strict_target_size && final_pool.size() > target_size) {
        auto t_trunc = std::chrono::high_resolution_clock::now();

        // Partition: initial dets first, new dets after
        fe::fe_set<DeterminantT<StorageType>, fe::DetHash<StorageType>> initial_set(
            initial_pool.begin(), initial_pool.end());

        std::vector<DeterminantT<StorageType>> kept, candidates;
        for (auto& d : final_pool) {
            if (initial_set.count(d)) {
                kept.push_back(std::move(d));
            } else {
                candidates.push_back(std::move(d));
            }
        }

        // Select candidates to keep: top-K by score, or uniform random
        if (kept.size() < target_size) {
            size_t n_add = std::min(candidates.size(), target_size - kept.size());

            if (mode == "uniform") {
                // Uniform random: shuffle candidates and take first n_add
                // Better diversity for systems with uniform integrals (e.g. Hubbard)
                std::mt19937 rng(42 + candidates.size());  // reproducible, varies per round
                std::shuffle(candidates.begin(), candidates.end(), rng);
                for (size_t i = 0; i < n_add; i++) {
                    kept.push_back(std::move(candidates[i]));
                }
            } else {
                // Score-based: extract scores, partial sort, take top-K
                std::vector<std::pair<double, size_t>> score_idx(candidates.size());
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < static_cast<int>(candidates.size()); i++) {
                    auto it = pool_map.find(candidates[i]);
                    double s = (it != pool_map.end()) ? std::abs(it->second) : 0.0;
                    score_idx[i] = {s, i};
                }

                std::nth_element(score_idx.begin(), score_idx.begin() + n_add,
                                 score_idx.end(),
                                 [](const auto& a, const auto& b) { return a.first > b.first; });

                for (size_t i = 0; i < n_add; i++) {
                    kept.push_back(std::move(candidates[score_idx[i].second]));
                }
            }
        }

        double t_trunc_s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_trunc).count();
        if (verbosity >= 1) {
            std::cout << "[PoolBuild] Strict truncation: " << pool_map.size()
                      << " -> " << kept.size()
                      << " (kept " << initial_set.size() << " initial), "
                      << std::fixed << std::setprecision(2) << t_trunc_s << "s" << std::endl;
        }
        final_pool = std::move(kept);
    }

    if (telemetry_out) {
        *telemetry_out = aggregate_telemetry;
    }

    if (filtered_growth_active && filtered_result_out) {
        // Milestone C2b-1: NEVER report pre-truncation counts -- recompute
        // composition from the actual returned final_pool, whether or not
        // strict truncation ran (with strict_target_size off, final_pool ==
        // the untruncated pool_map contents, and this recompute is still the
        // authoritative source of truth rather than the running counters).
        FilteredGrowthResult recomputed;
        recomputed.saturation_reason = filtered_result.saturation_reason;
        for (const auto& d : final_pool) {
            auto it = pool_labels.find(d);
            SectorLabel label = (it != pool_labels.end()) ? it->second : SectorLabel::Dispersion;
            switch (label) {
                case SectorLabel::Protected: recomputed.n_protected++; break;
                case SectorLabel::Dispersion:
                case SectorLabel::CtAllowed: recomputed.n_recovery++; break;
                case SectorLabel::LasAbsent: recomputed.n_aux++; break;
                default: break;
            }
        }
        recomputed.n_work = final_pool.size();
        recomputed.recovery_target_met = recomputed.n_recovery >= sector_constraint->recovery_target;
        recomputed.doorway_limit_reached = filtered_result.doorway_limit_reached;
        *filtered_result_out = recomputed;
    }

    return {final_pool, threshold};
}

// Wrapper for backward compatibility (pool_build)
// Note: uses heat_bath mode (default) for backward compatibility
std::pair<std::vector<Determinant>, double>
pool_build(
    const std::vector<Determinant>& initial_pool,
    const std::vector<double>& initial_coeff,
    int n_orb,
    const std::vector<std::vector<double>>& h1,
    const std::vector<double>& eri,
    double threshold,
    size_t target_size,
    HijCache& cache,
    const std::string& cache_file,
    int max_rounds,
    double threshold_decay,
    const std::vector<int>& attentive_orbitals,
    int verbosity
) {
    // Build params for backward compatibility
    PoolBuildParams params;
    params.max_rounds = max_rounds;
    params.threshold_decay = threshold_decay;
    // screening_mode defaults to "heat_bath"
    return pool_build_t<uint64_t>(initial_pool, initial_coeff, n_orb, h1, eri, 
                                  threshold, target_size, cache, cache_file, 
                                  attentive_orbitals, verbosity, params);
}

// Explicit instantiations
template std::vector<std::pair<DeterminantT<uint64_t>, double>>
process_parent_worker_t<uint64_t>(const DeterminantT<uint64_t>&, int, double, const DoubleExcTable&,
                                  const std::vector<std::vector<double>>&,
                                  const std::vector<double>&,
                                  const std::unordered_set<int>&,
                                  const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 2>>, double>>
process_parent_worker_t<std::array<uint64_t, 2>>(const DeterminantT<std::array<uint64_t, 2>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 3>>, double>>
process_parent_worker_t<std::array<uint64_t, 3>>(const DeterminantT<std::array<uint64_t, 3>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);

template std::pair<std::vector<DeterminantT<uint64_t>>, double>
pool_build_t<uint64_t>(const std::vector<DeterminantT<uint64_t>>&, std::vector<double>, int,
                       const std::vector<std::vector<double>>&,
                       const std::vector<double>&,
                       double, size_t, HijCacheT<uint64_t>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&, const IntegralSparsityInfo*, const DoubleExcTable*, const SectorConstraint<uint64_t>*, ScreeningTelemetry*, FilteredGrowthResult*);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 2>>>, double>
pool_build_t<std::array<uint64_t, 2>>(const std::vector<DeterminantT<std::array<uint64_t, 2>>>&, std::vector<double>, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 2>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&, const IntegralSparsityInfo*, const DoubleExcTable*, const SectorConstraint<std::array<uint64_t, 2>>*, ScreeningTelemetry*, FilteredGrowthResult*);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 3>>>, double>
pool_build_t<std::array<uint64_t, 3>>(const std::vector<DeterminantT<std::array<uint64_t, 3>>>&, std::vector<double>, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 3>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&, const IntegralSparsityInfo*, const DoubleExcTable*, const SectorConstraint<std::array<uint64_t, 3>>*, ScreeningTelemetry*, FilteredGrowthResult*);

template std::vector<std::pair<DeterminantT<std::array<uint64_t, 4>>, double>>
process_parent_worker_t<std::array<uint64_t, 4>>(const DeterminantT<std::array<uint64_t, 4>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 5>>, double>>
process_parent_worker_t<std::array<uint64_t, 5>>(const DeterminantT<std::array<uint64_t, 5>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 6>>, double>>
process_parent_worker_t<std::array<uint64_t, 6>>(const DeterminantT<std::array<uint64_t, 6>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 7>>, double>>
process_parent_worker_t<std::array<uint64_t, 7>>(const DeterminantT<std::array<uint64_t, 7>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);
template std::vector<std::pair<DeterminantT<std::array<uint64_t, 8>>, double>>
process_parent_worker_t<std::array<uint64_t, 8>>(const DeterminantT<std::array<uint64_t, 8>>&, int, double, const DoubleExcTable&,
                                                 const std::vector<std::vector<double>>&,
                                                 const std::vector<double>&,
                                                 const std::unordered_set<int>&,
                                                 const IntegralSparsityInfo*);

template std::pair<std::vector<DeterminantT<std::array<uint64_t, 4>>>, double>
pool_build_t<std::array<uint64_t, 4>>(const std::vector<DeterminantT<std::array<uint64_t, 4>>>&, std::vector<double>, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 4>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&, const IntegralSparsityInfo*, const DoubleExcTable*, const SectorConstraint<std::array<uint64_t, 4>>*, ScreeningTelemetry*, FilteredGrowthResult*);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 5>>>, double>
pool_build_t<std::array<uint64_t, 5>>(const std::vector<DeterminantT<std::array<uint64_t, 5>>>&, std::vector<double>, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 5>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&, const IntegralSparsityInfo*, const DoubleExcTable*, const SectorConstraint<std::array<uint64_t, 5>>*, ScreeningTelemetry*, FilteredGrowthResult*);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 6>>>, double>
pool_build_t<std::array<uint64_t, 6>>(const std::vector<DeterminantT<std::array<uint64_t, 6>>>&, std::vector<double>, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 6>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&, const IntegralSparsityInfo*, const DoubleExcTable*, const SectorConstraint<std::array<uint64_t, 6>>*, ScreeningTelemetry*, FilteredGrowthResult*);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 7>>>, double>
pool_build_t<std::array<uint64_t, 7>>(const std::vector<DeterminantT<std::array<uint64_t, 7>>>&, std::vector<double>, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 7>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&, const IntegralSparsityInfo*, const DoubleExcTable*, const SectorConstraint<std::array<uint64_t, 7>>*, ScreeningTelemetry*, FilteredGrowthResult*);
template std::pair<std::vector<DeterminantT<std::array<uint64_t, 8>>>, double>
pool_build_t<std::array<uint64_t, 8>>(const std::vector<DeterminantT<std::array<uint64_t, 8>>>&, std::vector<double>, int,
                                      const std::vector<std::vector<double>>&,
                                      const std::vector<double>&,
                                      double, size_t, HijCacheT<std::array<uint64_t, 8>>&, const std::string&, const std::vector<int>&, int, const PoolBuildParams&, const IntegralSparsityInfo*, const DoubleExcTable*, const SectorConstraint<std::array<uint64_t, 8>>*, ScreeningTelemetry*, FilteredGrowthResult*);

} // namespace trimci_core
