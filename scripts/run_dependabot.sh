#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY}"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "GITHUB_TOKEN is not set; cannot trigger Dependabot" >&2
    exit 1
fi
token="${GITHUB_TOKEN}"

for ecosystem in pip github-actions; do
  if ! curl --fail-with-body -S -s -X POST \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    -H "Content-Type: application/json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    https://api.github.com/repos/${repo}/dependabot/updates \
    -d "{\"package-ecosystem\":\"${ecosystem}\",\"directory\":\"/\"}"; then
    code=${PIPESTATUS[0]}
    if [[ $code -eq 22 ]]; then
      echo "Dependabot endpoint returned 404 for ${ecosystem}; skipping" >&2
      continue
    fi
    echo "Failed to trigger Dependabot for ${ecosystem}" >&2
    exit 1
  fi
done
