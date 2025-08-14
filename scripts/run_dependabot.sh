#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY}"
token="${GITHUB_TOKEN}"

for ecosystem in pip github-actions; do
  curl -X POST \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/${repo}/dependabot/updates \
    -d "{\"package-ecosystem\":\"${ecosystem}\",\"directory\":\"/\"}"
done
