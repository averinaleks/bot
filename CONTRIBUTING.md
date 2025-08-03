# Contributing

Thank you for your interest in contributing to this project!

## Development workflow

1. Create a virtual environment and install dependencies using the helper
   script:

    ```bash
    ./scripts/install-test-deps.sh
    ```
   (Use `--full` to include GPU packages.) This step is **required** before
   running the test suite—`pytest` expects these packages to be installed.
2. Run `python -m flake8` and the test suites before submitting a pull request.
   The CI workflow runs `flake8` as a dedicated step and will fail if any style
   errors are reported, so it's best to fix them locally first.

   ```bash
   pytest -m "not integration"  # unit tests
   pytest -m integration         # integration tests
   ```

   These tests are also executed automatically by `pre-commit`.

## CI troubleshooting

GitHub Actions occasionally fails to start a runner or download dependencies, reporting a temporary error. In most cases simply rerunning the workflow resolves the problem. If failures persist you can set up a [self-hosted runner](https://docs.github.com/actions/hosting-your-own-runners) to avoid queueing delays and network issues. For stubborn issues contact [GitHub Support](https://support.github.com/). Include the **correlation ID** from the failed run's `View raw logs` page so they can investigate.
