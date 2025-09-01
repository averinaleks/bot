#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
    echo "GITHUB_REPOSITORY is not set; cannot trigger Dependabot" >&2
    exit 1
fi
repo="${GITHUB_REPOSITORY}"

# Determine authentication token.
if [[ -n "${TOKEN:-}" ]]; then
    token="${TOKEN}"
elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
    token="${GITHUB_TOKEN}"
else
    echo "TOKEN or GITHUB_TOKEN is not set; export a token with repo and security_events scopes" >&2
    exit 1
fi

# Use --fail-with-body when supported to surface response bodies on errors.
fail_arg="--fail-with-body"
if ! curl --help | grep -q -- '--fail-with-body'; then
  fail_arg="--fail"
fi

for ecosystem in pip github-actions; do
  set +e
  response=$(curl ${fail_arg} -S -s -X POST \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    -H "Content-Type: application/json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${repo}/dependabot/updates" \
    -d "{\"package-ecosystem\":\"${ecosystem}\",\"directory\":\"/\"}")
  code=$?
  set -e
  if [[ $code -eq 22 ]]; then
    echo "Dependabot endpoint returned 404 for ${ecosystem}" >&2
    echo "${response}" >&2
    continue
  fi
  if [[ $code -ne 0 ]]; then
    echo "Failed to trigger Dependabot for ${ecosystem}" >&2
    echo "${response}" >&2
    exit 1
  fi
done
