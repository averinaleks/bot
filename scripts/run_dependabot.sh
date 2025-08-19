#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY}"
token="${GITHUB_TOKEN}"

for ecosystem in pip github-actions; do
  if ! curl -f -S -s -X POST \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    -H "Content-Type: application/json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    https://api.github.com/repos/${repo}/dependabot/updates \
    -d "{\"package-ecosystem\":\"${ecosystem}\",\"directory\":\"/\"}"; then
    echo "Failed to trigger Dependabot for ${ecosystem}" >&2
    exit 1
  fi
done
