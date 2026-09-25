<p align="center">
  <img src="https://raw.githubusercontent.com/hao-zhang-quantum/TrimCI/44341c9d1a0cd070ed9c06ad4b1e437a54358514/banner.png"
       alt="TrimCI banner"
       width="800"/>
</p>

# TrimCI

> **This is the FRASCI-specific development fork of TrimCI.** It is used by the FRASCI
> (FRAgment-based Selected CI) framework's `frasci.parent`/`frasci.recovery` packages,
> and is based on upstream [`hao-zhang-quantum/TrimCI`](https://github.com/hao-zhang-quantum/TrimCI)
> (this repository's `upstream` remote), from which it retains full commit history.
> These modifications have not been proposed to or reviewed by the upstream maintainer
> yet, so this repository must **not** be presented as an official upstream TrimCI
> release. Original authorship and the MIT license (below) remain unchanged. FRASCI
> releases pin an exact commit of this repository for reproducibility — see
> "FRASCI-specific changes" below.

> 🔥 **v0.2.0 (May 2026):** Now supports **orbital optimization** and **fast expansion**. The complete three-phase workflow is exposed via the friendly `trimci.ground_state(...)` entry, or the production-grade `TrimCI_skill.py` template. Just `pip install trimci`.

**Trimmed Configuration Interaction (TrimCI)** is a **high-performance** framework for quantum many-body and quantum chemistry calculation.
It constructs accurate ground states *directly from random Slater determinants* — without any guiding ansatz, Hartree–Fock reference, or prior human knowledge — through an iterative expansion–trimming cycle on the determinant graph.

TrimCI demonstrates that accurate many-body ground states can *emerge from randomness*, achieving state-of-the-art accuracy and efficiency across molecular and lattice systems. It can **outperform human-designed ansatzes or human-provided knowledge** in hard problems, such as strongly correlated systems.

**Papers**

- H. Zhang, M. Otten, “From Random Determinants to the Ground State,” arXiv:2511.14734 (2025). https://arxiv.org/abs/2511.14734
- H. Zhang, M. Otten, “Absorbing Many-Body Correlations into Core-Optimized Orbitals,” arXiv:2605.22977 (2026). https://arxiv.org/abs/2605.22977

---

## 🚀 Install

```
pip install trimci
```

Alternatively, you may build the package on your environment `python -m pip install .`.

## 🤖 For AI agents

**TrimCI is designed to be LLM-agents-friendly.** [`py/trimci/TrimCI_skill.py`](py/trimci/TrimCI_skill.py) documents the three-phase workflow and the rationale for each parameter, so coding agents can drive TrimCI calculations on their own.

> 📖 **For agents:** read [`py/trimci/TrimCI_skill.py`](py/trimci/TrimCI_skill.py). It documents the three-phase workflow (Phase 0 Core Search + OrbOpt → Phase 1 Expansion + OrbOpt → Phase 2 Final Expansion) with full parameter references and tuning guidance.

## ⚡ Quick Tutorial

### A. Easy-to-use entry point — `ground_state`

Three parameters, one call:

```python
import trimci

# FCIDUMP path
result = trimci.ground_state("FCIDUMP", n_dets=10000)
print(result.energy)

# Or directly from a PySCF mean-field
result = trimci.ground_state(mf, n_dets=10000)
```

Full walkthrough in [`tutorials/complete_workflow_tutorial.ipynb`](tutorials/complete_workflow_tutorial.ipynb); core-search component in [`tutorials/core_search_tutorial.ipynb`](tutorials/core_search_tutorial.ipynb).

📘 Or browse the [**HTML reference**](https://raw.githack.com/hao-zhang-quantum/TrimCI/main/tutorials/ground_state_tutorial.html) directly in your browser.

### B. State-of-the-art template — `TrimCI_skill.py`

To push toward state-of-the-art accuracy, copy [`py/trimci/TrimCI_skill.py`](py/trimci/TrimCI_skill.py) as a starting template, edit the configuration for your system, and run. The script is self-contained and exposes every tuning knob.

## ✨ Key Features

- **Emergent accuracy from randomness:** discovers the ground state without predefined ansatz or human bias.
- **Expansion–trimming mechanism:** iteratively expands the determinant space via Hamiltonian couplings and trims away unimportant configurations.
- **C++ backend, Python interface:** efficient C++ backend with OpenMP parallelization for core functions, while Python interface provides user-friendly access.
- **Massive efficiency gain:** achieves equivalent accuracy to selected-CI using orders-of-magnitude fewer determinants (see Scientific Highlights).
- **Transferable module:** TrimCI wavefunctions can initialize or guide AFQMC, VMC, DMRG, tensor networks, and quantum algorithms (VQE, QPE).
- **Explicit wavefunction:** produces a compact, analyzable coefficients and determinants dict enabling direct evaluation of observables and other measures.

---

## 🧱 Three-Phase Workflow

`ground_state` (and `TrimCI_skill.py`) runs a production-grade three-phase pipeline:

1. **Phase 0 — Core Search + Orbital Optimization.**
   Multi-run stochastic exploration finds a high-quality initial determinant space; orbital optimization rotates the basis for compactness.
2. **Phase 1 — Expansion + Orbital Refinement.**
   Grow the determinant space while co-evolving orbitals at every step, avoiding the convergence slowdown caused by frozen orbitals.
3. **Phase 2 — Final Expansion.**
   Freeze orbitals, rapidly expand to the target determinant count, optionally with PT2 correction.

**Phase 0 Core Search** is an **expansion–trimming cycle** on the determinant graph (nodes = Slater determinants, edges = Hamiltonian couplings \(H_{ij}\)):

- **Expansion:** add neighboring determinants with large couplings \(|H_{ij}c_j|>\theta\).
- **Trimming:** local random-group diagonalizations remove negligible states; a global merge then selects top-amplitude survivors.

Phases 1 and 2 then drive **fast expansion** from this high-quality core, refining the variational subspace nearly monotonically toward the ground state.

---

## 🧠 Scientific Highlights

- **Molecular systems:**
  Matches SHCI accuracy on Cr₂, [4Fe–4S], and the nitrogenase P-cluster while using \(10^2\)–\(10^5\times\) fewer determinants.
- **Lattice systems:**
  For the 8×8 Hubbard model, TrimCI reproduces >99 % of the AFQMC ground-state energy using only \(10^{-28}\) of the Hilbert space. On 4×4 lattices, TrimCI achieves higher accuracy than AFQMC benchmarks.
- **Emergent structure:**
  Starting from random determinants, TrimCI self-organizes a compact “core set” of dominant configurations.
  The amplitude distribution follows a power law \(p(r) \propto r^{-(1+\alpha)}\) , revealing a *scale-free* organization and quantifiable **algorithmic entropy**.

---

## 🔗 Integration with Other Frameworks

TrimCI provides a compact and explicit coefficients and determinants dict that can:

- serve as a **trial or guiding wavefunction** for AFQMC and VMC,
- initialize **DMRG** and tensor-network optimizations,
- provide high-overlap initial states for **VQE** or **QPE** quantum algorithms,
- enable cross-validation and hybrid workflows across classical and quantum domains.

---

## 🧬 FRASCI-specific changes

Already on this fork's committed `main` branch (built on upstream `v0.2.0`):

1. **Native recovery policy and sector-constraint support** (`RecoveryPolicy`,
   `SectorConstraint`, `make_sector_constraint_64`/`_128`) — lets a caller protect a
   fragment-derived determinant set and grow only a bounded, classified pool of
   additional (same-sector/dispersion/charge-transfer) determinants around it.
2. **Phased native expansion** (`ExpansionConfig`, `run_expansion_phased`/`_128`) — the
   multi-phase Phase-0/1/2 expansion entry point the parent/recovery workflows above
   drive directly, rather than the single-phase `ground_state` wrapper.

Added on top, currently uncommitted in this checkout (staged for the same commit as the
two items above):

3. **`ExpansionConfig.allowed_orbital_pairs`** — restricts BFGS orbital-pair rotation to
   a caller-supplied allowlist of `(p, q)` index pairs, so an orbital optimization can be
   confined to (for example) intra-fragment pairs and preserve a requested orbital gauge
   instead of rotating freely across the whole space.
4. **`seed_trim_rngs(seed)`, plus Python-level seed propagation** — reseeds the
   previously timestamp-only, unreseedable file-scope RNG streams behind partition
   shuffling and Davidson initialization, so independent CPU restarts (and CPU-side
   partitioning) become reproducible given a fixed seed. Previously
   `IterativeWorkflowParams.random_seed` was accepted and stored but never actually read
   on the CPU single/multi-run path.
5. **Validation of `allowed_orbital_pairs`** — rejects out-of-range orbital indices and
   duplicate pairs before an optimization run starts, rather than failing (or
   silently misbehaving) partway through.

## 📦 Install (this fork)

```bash
python -m pip install "trimci @ git+https://github.com/Abhi-Muvva/TrimCI-FRASCI.git@main"
```

FRASCI itself pins a **full commit SHA** of this repository, not a branch name, for
reproducibility (see FRASCI's own `pyproject.toml`/`requirements.txt`). The branch
install above is only a convenience for exploring this fork directly. Installing from a
private GitHub repository requires an authenticated `git`/`pip` setup for
`github.com` — `gh auth login` (HTTPS) or an SSH key registered to your GitHub account
both work.

## 📚 Version history

See [`CHANGELOG.md`](CHANGELOG.md).

## 📜 License

MIT License — see [`LICENSE`](LICENSE) for details.
