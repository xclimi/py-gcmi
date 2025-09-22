# Project Plan — py-gcmi

Status: Draft
Version: 0.1
Last Updated: 2025-09-22

Scope: Establish a minimal, testable, and well-documented Python reference implementation of GCMI Core 0.1 with an initial middleware/Hook ecosystem; provide an experimental SpectralOps integration path.

1. Goals and Non‑Goals

- Goals
  - Implement GCMI Core 0.1: init_fn, step_fn, run_fn, fully type-hinted and backend-neutral (NumPy/JAX/Torch via xp facade).
  - Provide a minimal but valuable middleware set (stability, limiting, positivity, conservation, energy fix).
  - Provide hooks for observability (energy/water budgets, timing, simple IO).
  - Define ops facade (grid ops; optional spectral ops) with clean capability negotiation.
  - Ship unit tests, baseline tests, and CI per AGENTS.md (ruff, black, isort, mypy --strict, pytest).
  - Provide runnable examples for quick validation.
  - English-first documentation and ADRs for major decisions.

- Non‑Goals (for v0.1)
  - Full production readiness or large-scale distributed execution.
  - Broad physics library beyond minimal tendencies suitable for examples.
  - Rich visualization/plotting stack (keep minimal and optional).

2. Milestones and Timeline (indicative)

- M0 (Week 0): Approve plan & design docs
- M1 (Week 2): Core 0.1 APIs + grid ops + minimal middleware + hooks + baseline tests
- M2 (Week 4): Config schema + examples + CI (ruff, black, isort, mypy, pytest) green
- M3 (Week 6): SpectralOps facade + spectral dynamics middleware (NumPy + shtns prototype; experimental)
- M4 (Week 8): Docs polished; ADRs for key choices; tag py-gcmi 0.1.0

3. Deliverables per Milestone

- M1
  - gcmi/core/api.py: init_fn, step_fn, run_fn; thin physics_core skeleton
  - gcmi/ops/grid.py minimal operators (diffusion/Laplacian placeholders, metric access)
  - Middleware: with_cfl_guard, with_hyperdiff, with_flux_limiter, with_positivity, with_energy_fix, with_conservation_projection
  - Hooks: energy_budget, water_budget, timer (CSV/NDJSON outputs)
  - Baseline tests: conservation (dry/no forcing), stability smoke, reproducibility (fixed RNG)

- M2
  - Config loader and schema (env → cfg bridge, strict validation)
  - Examples: dry dynamics (grid), moist toy case
  - CI: pre-commit hooks; GitHub Actions enforcing AGENTS.md toolchain
  - Documentation: README quick start, basic user/developer guides

- M3 (Experimental)
  - ops/spectral facade (protocol + capability marker)
  - with_spectral_dyn_core (pseudo-spectral path; θ-method + hyperviscosity + dealiasing)
  - Spectral diagnostics hook (energy spectrum, enstrophy)
  - Example: T42 channel/dry core run

- M4
  - Documentation polish: this plan, system design, CONTRIBUTING, ADRs
  - “Supports GCMI Core 0.1” declaration; SpectralOps 0.1 marked experimental
  - py-gcmi 0.1.0 release tag

4. Acceptance Criteria

- API: All public code has complete type hints; mypy --strict passes.
- Quality: ruff, black, isort clean; pre-commit installed and green.
- Tests: pytest baseline green; non-flaky; pytest-cov reports maintained.
- Stability: CFL guard effective; positivity preserved for configured vars; hyperdiff/limiters work as configured.
- Energy/water: energy_fix and conservation projection keep drift in baseline within epsilon budget; hooks produce interpretable CSV.
- Examples: end-to-end runs complete with documented commands.
- Spectral (experimental): forward/inverse roundtrip tolerance; Helmholtz solve validated; T42 run completes without blow-up.

5. Risks and Mitigations

- Numerical instability on varied grids
  - Mitigation: conservative default chain order and coefficients; baseline tests + hooks.
- Backend divergence (numpy/jax/torch)
  - Mitigation: xp facade and ops indirection; limit backend-specific code to adapters.
- Spectral dependency availability (shtns)
  - Mitigation: optional dependency; graceful degrade to grid-only.
- Scope creep
  - Mitigation: ADR gate for features; milestone stickiness; small atomic PRs.

6. Roles and Communication

- Maintainers: code review, CI, tagging, ADR stewardship, releases.
- Contributors: follow AGENTS.md protocols; small focused PRs with tests.
- Rituals: lightweight weekly milestone sync; PR templates with “what/why” and links to ADRs.

7. Versioning and Release

- Semantic versioning for py-gcmi; explicit “Supports GCMI Core x.y” in docs.
- v0.1.0 targets Core 0.1; SpectralOps 0.1 experimental.
- CHANGELOG.md maintained using “Keep a Changelog”.

8. Success Metrics

- CI green on main; pre-commit enforced.
- Baseline tests stable across runs and backends (within tolerances).
- Middleware interoperability demonstrated in examples; hooks produce budgets.
- Developer onboarding time: ≤ 30 minutes to run an example from a clean clone.

9. Out of Scope (v0.1)

- Distributed drivers (e.g., JAX pjit, Torch DDP, MPI) beyond structure placeholders.
- Full-featured NetCDF/Zarr pipelines; advanced plotting.
- Comprehensive physics packages (only minimal tendencies for examples).

10. Dependencies and Tooling

- Python ≥ 3.10; dependency management via uv.
- Tooling: ruff, black, isort, mypy, pytest (+ pytest-cov), pre-commit.
- Optional: shtns or alternative spectral libraries for experimental SpectralOps.

11. Workstreams and Tracking

- Core API & ops (owner: Core)
- Middleware & hooks (owner: Numerics)
- Config & examples (owner: DX)
- Spectral extension (owner: Spectral)
- CI & docs & ADRs (owner: Infra)
- Each workstream publishes checklists in docs/notes or GitHub project boards.

12. Deliverable Artifacts

- Source code under gcmi/ with typed public APIs.
- Tests under tests/ with baseline datasets/fixtures.
- Docs under docs/: plan.md, design.md, ADRs, guides.
- Examples under examples/ runnable via documented commands.

Appendix: Status Labels

- Draft → In Review → Approved → Implemented → Deprecated
- All docs/pages should carry a visible status label to reflect maturity.
