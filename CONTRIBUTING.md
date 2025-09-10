# Contributing

Thank you for your interest in contributing to this project!

## Development workflow

1. Create a virtual environment, install dependencies, and set up the test
   tooling:

    ```bash
    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt -r requirements-dev.txt
    ./scripts/install-test-deps.sh
    ```
   (Use `--gpu` with the script to include GPU packages.) This step is
   **required** before running the test suite—`pytest` expects these packages to
   be installed.
2. Run the style checks and test suites before submitting a pull request:

    ```bash
    pytest -m "not integration"
    pytest -m integration
    flake8 . && pip-audit
    ```
   The CI workflow runs these checks and will fail if any issues are reported,
   so it's best to fix them locally first.

### Running tests

Use the following commands to execute the test suites locally:

```bash
pytest -m "not integration"   # unit tests
pytest -m integration         # integration tests
```

These tests are also executed automatically by `pre-commit`.

### Updating dependencies

The dependency lists are managed with [`pip-compile`](https://github.com/jazzband/pip-tools).
Edit `requirements-core.in` or `requirements-gpu.in` to change top-level
dependencies and regenerate the pinned files:

```bash
pip-compile --strip-extras requirements-core.in        # updates requirements-core.txt
pip-compile --strip-extras requirements-gpu.in       # updates requirements-gpu.txt
```

We pass `--strip-extras` explicitly so that updating `pip-tools` doesn't change whether extras appear in the generated files.

Commit both the `.in` and generated `.txt` files when updating.

## Repository secrets

Some GitHub Actions workflows pull container images from GitHub Container Registry
(GHCR) and trigger Dependabot runs. These workflows expect a secret named
`TOKEN` that provides a personal access token with the `repo` and
`security_events` scopes so the runner can authenticate and interact with the
GitHub API.

## CI troubleshooting

GitHub Actions occasionally fails to start a runner or download dependencies, reporting a temporary error. In most cases simply rerunning the workflow resolves the problem. If failures persist you can set up a [self-hosted runner](https://docs.github.com/actions/hosting-your-own-runners) to avoid queueing delays and network issues. For stubborn issues contact [GitHub Support](https://support.github.com/). Include the **correlation ID** from the failed run's `View raw logs` page so they can investigate.
