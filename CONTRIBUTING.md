# Contributing

Thank you for your interest in contributing to this project!

## Development workflow

1. Install dependencies in a virtual environment using the helper script:

   ```bash
   ./scripts/install-test-deps.sh
   ```
2. Run `flake8` and `pytest` before submitting a pull request. These tests are
   also executed automatically by `pre-commit`.

## CI troubleshooting

GitHub Actions occasionally fails to start a runner or download dependencies, reporting a temporary error. In most cases simply rerunning the workflow resolves the problem. If failures persist you can set up a [self-hosted runner](https://docs.github.com/actions/hosting-your-own-runners) to avoid queueing delays and network issues. For stubborn issues contact [GitHub Support](https://support.github.com/). Include the **correlation ID** from the failed run's `View raw logs` page so they can investigate.
