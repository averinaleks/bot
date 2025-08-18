# Contributing

Thank you for your interest in contributing to this project!

## Development workflow

1. Create a virtual environment and install dependencies using the helper
   script:

    ```bash
    ./scripts/install-test-deps.sh
    ```
   (Use `--gpu` to include GPU packages.) This step is **required** before
   running the test suite—`pytest` expects these packages to be installed.
2. Run `python -m flake8` and the test suites before submitting a pull request.
   The CI workflow runs `flake8` as a dedicated step and will fail if any style
   errors are reported, so it's best to fix them locally first.

### Running tests

Use the following commands to execute the test suites locally:

```bash
pytest                        # unit tests (integration tests skipped)
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
(GHCR). These workflows expect a secret named `TOKEN` that provides a personal
access token with the `read:packages` scope so the runner can authenticate and
download images from GHCR.

## CI troubleshooting

GitHub Actions occasionally fails to start a runner or download dependencies, reporting a temporary error. In most cases simply rerunning the workflow resolves the problem. If failures persist you can set up a [self-hosted runner](https://docs.github.com/actions/hosting-your-own-runners) to avoid queueing delays and network issues. For stubborn issues contact [GitHub Support](https://support.github.com/). Include the **correlation ID** from the failed run's `View raw logs` page so they can investigate.
