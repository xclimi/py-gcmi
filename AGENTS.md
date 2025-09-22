# Engineering Protocol for Python Agent Development

### A Guide for AI and Human Contributors

## 1. Mission Briefing

This document outlines the engineering principles and development protocols applicable to this project and to modern Python development in general. As a contributor to this codebase, whether AI or human, you **are required to** adhere to these standards.

Our primary objective is not merely to write code that runs, but to build a system that is robust, maintainable, performant, and reliable. Your success as a contributor will be measured by your disciplined adherence to these protocols.

## 2. Core Principles

1.  **Elegant design**: Strive for simplicity and elegance in your solutions. Avoid over-engineering; the simplest solution that meets the requirements is often the best. Follow established design patterns and best practices.
2.  **Clarity of code**: Write code that is easily understood by the next contributor.Avoid obscure language features or overly complex one-liners. The Zen of Python (`import this`) is our guiding philosophy. Code is read far more often than it is written.
2.  **Type Safety and Explicitness**: All new code **shall** be fully type-hinted. Static type checking is not optional; it is a fundamental tool for preventing entire classes of bugs and improving code comprehensibility.
3.  **Test Rigorously**: A feature without tests is an incomplete feature. Untested code is considered legacy or broken by default. Tests are a form of living documentation and a safety net for future changes.
4.  **Automate Everything**: All formatting, linting, type checking, testing, and deployment processes **shall** be automated through a CI/CD pipeline. Human intervention in these processes should be the exception, not the rule.
5.  **Document Diligently**: While code should be as self-documenting as possible, this does not replace formal documentation. Your changes **must** be accompanied by clear explanations, docstrings, and updates to project-level documentation where necessary.
6.  **Design Before Code**: Significant new features or architectural changes require a concise Architecture Decision Record (ADR). This practice forces clear thinking upfront and serves as a historical record of the project's evolution.
7.  **Principle of Focused, Atomic Changes**: Avoid large, monolithic refactors. Changes should be small, focused, and submitted as atomic commits within a single Pull Request. Each PR should do one thing and do it well.

## 3. Technical Protocols

### 3.1. Code Quality & Style

* **Formatting**: All Python code **must** be formatted using `black` with its default configuration.
* **Linting**: All Python code **must** pass `ruff` checks without any warnings. Custom rules may be defined in `pyproject.toml`.
* **Import Sorting**: All Python imports **must** be sorted using `isort` (project defaults). Keep grouping and ordering deterministic.
* **Type Checking**: All Python code **must** pass static analysis by `mypy` in strict mode (`--strict`).
* **Automation**: These checks (**ruff**, **black**, **isort**, **mypy**, **pytest**) **must** be automated locally using the `pre-commit` framework. Any commit that fails pre-commit hooks will be rejected by the CI pipeline. Provide convenience scripts/Make targets where helpful.
* **Modularity**: Functions and classes should be small and adhere to the Single Responsibility Principle (SRP).
* **Naming**: Follow the PEP 8 naming conventions. Use descriptive, unambiguous names for variables, functions, and classes.

#### 3.1.1 Mapping ergonomics (dict "destructuring")

- Prefer `operator.itemgetter` or `gcmi.utils.struct.take/require` for required keys (concise, fast, explicit KeyError on missing keys).
- For nested access, use `gcmi.utils.struct.take_nested` with dotted paths in validators, config assembly, and middleware wiring.
- Structural pattern matching (`match`/`case`, Python 3.10+) is allowed where it clearly improves readability or you need to bind the “rest” via `**rest`. Avoid in tight loops/hot paths.
- Avoid ad‑hoc "dotdict" wrappers; they degrade static checking and may hide missing keys. Use `TypedDict`/`NamedTuple` views if helpful, while keeping runtime objects as plain dicts.
- Do not introduce silent defaults at call sites. Defaults belong in the validated config schema.

Examples:

```python
from operator import itemgetter as ig
from gcmi.utils.struct import take, require, take_nested

# Required flat keys (fast path)
T, q, u, v = take(state, "T", "q", "u", "v")

# Better error on missing keys
(dx_min,) = require(params["grid"], "dx_min")

# Nested access with dotted paths
dx_min, theta = take_nested(params, "grid.dx_min", "spectral.semi_implicit.theta")

# Structural pattern matching when readability matters
match state:
    case {"T": T, "q": q, "u": u, "v": v, **rest}:
        ...
```

#### 3.1.2 Declarative runtime requirements (decorator + early-check middleware)

Use a small, explicit contract to declare what a component needs from `state` / `params` / `forcing`, and validate only in the first few steps for minimal overhead later.

- API (gcmi.utils.requirements)
  - `Requirement(where, path, required=True, type=None, predicate=None, severity="error")`
    - `where`: `"state" | "params" | "forcing"`
    - `path`: dotted path, e.g. `"spectral.radius"` or `"grid.dx_min"`
    - `type`: `isinstance` check (type or tuple of types)
    - `predicate`: `callable(value) -> bool`, must return True
    - `severity`: `"error"` (default) or `"warn"`
  - `@requires(*Requirement)`: attach requirements to a StepFn or callable
  - `get_requirements(fn)`: retrieve attached requirements (follows `__wrapped__`)
  - `validate_requirements(...)`: validate against containers, returns `(errors, warnings)`
- Middleware (gcmi.middleware.requirements)
  - `with_requirements_check(step, *, max_checks=3, raise_on_error=True, record_warnings=True)`
  - Validates attached (+extra) requirements only for the first `max_checks` invocations; then skips checks entirely.

Example (SpectralOps: require params.spectral.radius > 0)
```python
from gcmi.utils.requirements import Requirement, requires
from gcmi.middleware.requirements import with_requirements_check

@requires(Requirement("params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0.0))
def step_core(state, forcing, params, dt, *, xp):
    # physics/dynamics tendencies...
    return state, {}

# Enforce only in early steps (e.g., first 3); afterwards skip for performance.
step = with_requirements_check(step_core, max_checks=3, raise_on_error=True)
```

Guidelines
- Prefer `@requires` at definition sites of step functions/middleware that need structural inputs (e.g., spectral adapters expect `params.spectral.radius`).
- Drivers/examples should wrap the assembled `step` with `with_requirements_check(...)` to fail fast at startup while keeping steady-state cost zero.
- Keep predicates side-effect free and simple; use `type` and `required` where possible.
- On violation with `raise_on_error=False`, the wrapper records a summary under `diag["gcmi_requirements"]` for audit.

### 3.2. Testing Protocol (Mandatory)

* **Framework**: `pytest` is the standard testing framework for this project.
* **Unit Tests**: Must cover the happy path, edge cases, and error conditions for individual components. Mocking should be used judiciously to isolate units under test.
* **Integration Tests**: Must verify that different parts of the application collaborate correctly. This includes API endpoints, data pipelines, and interactions with external services, using dedicated test datasets.
* **Coverage**: While we do not enforce a strict percentage, all new code must be accompanied by tests, and Pull Requests that decrease test coverage will be scrutinized. Use `pytest-cov` to monitor coverage.

### 3.3. Documentation Protocol

**Guiding Principle**: Documentation is a critical component of the software, not an afterthought. It must be created and maintained with the same rigor as the code. Our documentation is categorized into three primary types:

1.  **User-Facing Documentation**
    * **Purpose**: To enable end-users to effectively understand, install, and use the software/service.
    * **Components**:
        * `README.md`: Must contain a clear project overview, installation steps, and quick-start examples.
        * `CHANGELOG.md`: Must be kept current following the "Keep a Changelog" format, detailing all user-visible changes for each release.
        * **User Guides & Tutorials**: For complex features, dedicated guides or tutorials should be created (e.g., in a `/docs` directory).
        * **API Reference**: For any public APIs, a clear, complete, and preferably auto-generated reference (e.g., via Sphinx/MkDocs from docstrings) is mandatory.

2.  **Developer-Facing Documentation**
    * **Purpose**: To enable current and future developers to efficiently understand, maintain, and contribute to the codebase.
    * **Components**:
        * **Docstrings**: All public modules, functions, classes, and methods **must** have docstrings compliant with PEP 257 (Google or NumPy style recommended). These serve as the source for auto-generated internal API documentation.
        * `CONTRIBUTING.md`: A dedicated file explaining how to set up the development environment, run tests, adhere to style guides, and submit a pull request.
        * **Architectural Documentation**: Includes diagrams, Architecture Decision Records (ADRs), and explanations of core components and internal workflows.

3.  **Project/Feature Design Documentation**
    * **Purpose**: To define the scope, technical design, implementation plan, and risks for new features or significant changes *before* implementation begins. This directly aligns with our "Design Before Code" core principle.
    * **Workflow**:
        1.  **Proposal**: A design document (such as an ADR or a technical specification) **must** be created and reviewed before development.
        2.  **Status Tracking**: The document's status (e.g., `Draft`, `In Review`, `Approved`, `Implemented`, `Deprecated`) **must** be clearly marked and kept up-to-date with the project's progress.
        3.  **Post-Implementation Integration**: Upon completion of the feature, relevant information from the design (e.g., new API endpoints, architectural changes) **must** be integrated into the user-facing and developer-facing documentation. This ensures the internal consistency of the entire documentation system. The design document is then archived as a historical record.

### 3.4. Version Control (Git) Protocol

* **Authorization**: You are authorized to use `git` and the GitHub CLI (`gh`) for all repository interactions.
* **Branching**: All work **must** be done on feature branches created from the `main` branch. Direct commits to `main` are prohibited. Branch names should be descriptive (e.g., `feat/add-user-authentication`, `fix/resolve-caching-bug`).
* **Commit Messages**: Commits **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This is essential for automated versioning and changelog generation.
* **Pull Requests (PRs)**: Your PR description **must** clearly explain the "what" (what was done) and the "why" (the business or technical reason for the change). Link to relevant issues or ADRs.

### 3.5. Error Handling Protocol

* **No Silent Failures**: Functions **must not** silently ignore errors (e.g., via a bare `except:` block). Fail loudly and explicitly.
* **Custom Exceptions**: A well-defined hierarchy of custom exceptions is encouraged to provide specific, actionable error information to callers.
* **Clarity**: Error messages should be clear, concise, and provide context to help with debugging.

### 3.6. Performance Protocol

* **Benchmarking**: Use `pytest-benchmark` for performance-critical functions to quantify the impact of changes.
* **No Premature Optimization**: "Premature optimization is the root of all evil." First, make it work. Then, make it right. Only then, if necessary, make it fast. Performance improvements **must** be driven by profiling and identifying actual bottlenecks.

## 4. Operational Protocols

### 4.1. Dependency & Environment Management

* **Dependency Manager**: `uv` is the standard tool for managing dependencies and virtual environments.
* **Environment Isolation**: All development **must** occur within a virtual environment to prevent dependency conflicts.
* **Dependency Declaration**: All project dependencies **must** be explicitly declared in the `[project.dependencies]` section of `pyproject.toml`. Lock files generated by `uv` ensure reproducible builds.
* **Python Version**: The project's required Python version **shall** be pinned (e.g., via a `.python-version` file for `pyenv`) to ensure consistency across all development and production environments.

### 4.2. Logging Protocol

* **Library**: Use the standard `logging` library.
* **Structured Logging**: Logs should be structured (e.g., JSON format) in production environments to facilitate parsing and analysis by monitoring tools.
* **Levels**: Use appropriate log levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). Avoid excessive logging, especially at the `INFO` level.
* **Content**: Log messages should include a timestamp, log level, module name, and a descriptive message. Avoid logging sensitive information.

### 4.3. Configuration Management

* **Source**: Configuration should be loaded from environment variables or dedicated configuration files (`.toml`, `.yaml`), never hardcoded.
* **Validation**: Use a library like `pydantic` to define, validate, and manage application configuration. This provides type safety and clear error messages for invalid configurations.
* **Secrets Management**: Sensitive information (API keys, passwords) **must never** be committed to the repository. Use a secrets management system (e.g., HashiCorp Vault, AWS Secrets Manager) or environment variables.

### 4.4. Security Protocol

* **Dependency Scanning**: Regularly scan dependencies for known vulnerabilities using tools like `safety` or `pip-audit`. This **shall** be part of the CI pipeline.
* **Static Analysis**: Employ Static Application Security Testing (SAST) tools to identify potential security flaws in the codebase.
* **Principle of Least Privilege**: The application should run with the minimum permissions necessary to perform its function.

### 4.5. Code Review Protocol

* **Mandatory Review**: All PRs **must** be reviewed and approved by at least one other developer before being merged.
* **Review Focus**: Reviews should be constructive and focus on code quality, correctness, test coverage, documentation, and adherence to these protocols.
* **Author's Responsibility**: The PR author is responsible for ensuring their code is clean, tested, and easy to review. They are also expected to respond to feedback in a timely manner.
* **Reviewer's Responsibility**: The reviewer is responsible for providing thorough, respectful, and actionable feedback. The goal is collaborative improvement of the codebase, not personal criticism.

## 5. Engineering Checklist (Pre-merge)

Before merging any change, complete the following:

- Code Quality & Tests
  - [ ] Run ruff with autofix: `ruff check . --fix`
  - [ ] Run black formatting: `black .`
  - [ ] Run import sorting: `isort .`
  - [ ] Run tests: `pytest -q` (and ensure non-flaky, meaningful coverage)
  - [ ] (If applicable) Run mypy: `mypy --strict .` and resolve typing issues
  - [ ] Ensure pre-commit hooks are installed and passing locally

- Documentation Updates
  - [ ] Update README.md if user-facing behavior or commands changed
  - [ ] Update /docs (user + technical) where relevant:
        - User docs (e.g., data handling, training, inference, evaluation)
        - Technical docs (e.g., extension guides, specifications)
  - [ ] Update docs/index.md navigation to include any new docs pages
  - [ ] Add code docstrings for new/changed public APIs

- Project Hygiene
  - [ ] Update CHANGELOG.md with user-visible changes (Keep a Changelog format)
  - [ ] If introducing new conventions or tools, update AGENTS.md accordingly
  - [ ] Verify CI configuration enforces ruff, black, isort, mypy, pytest
  - [ ] Consider version bump if public API/CLI changes

- PR Quality
  - [ ] Reference related ADRs/Project docs (e.g., Projects/00x_*.md)
  - [ ] Provide a clear PR description covering the "what" and "why"
  - [ ] Keep changes focused and atomic; avoid unrelated refactors
