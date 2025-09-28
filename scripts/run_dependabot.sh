#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GITHUB_REPOSITORY:-}" ]]; then
    echo "GITHUB_REPOSITORY is not set; cannot trigger Dependabot" >&2
    exit 1
fi
repo="${GITHUB_REPOSITORY}"

token="${TOKEN:-${GITHUB_TOKEN:-}}"
if [[ -z "${token}" ]]; then
    echo "TOKEN or GITHUB_TOKEN is not set; export a PAT with repo and security_events scopes" >&2
    exit 1
fi

# Store all temporary responses in an isolated directory so that mktemp never
# writes into shared world-writable locations. This avoids Semgrep's
# ``shell.lang.security.insecure-mktemp`` finding and guarantees cleanup on
# abnormal exits.
tmp_dir="$(mktemp -d -t run_dependabot.XXXXXX)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

for ecosystem in pip github-actions; do
  tmp_response="$(mktemp "${tmp_dir}/${ecosystem}.XXXXXX")"
  http_status=$(curl -S -s -o "${tmp_response}" -w "%{http_code}" -X POST \
    -H "Authorization: Bearer ${token}" \
    -H "Accept: application/vnd.github+json" \
    -H "Content-Type: application/json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${repo}/dependabot/updates" \
    -d "{\"package-ecosystem\":\"${ecosystem}\",\"directory\":\"/\"}")
  if [[ ${http_status} -eq 404 ]]; then
    echo "Dependabot endpoint returned 404 for ${ecosystem}" >&2
    cat "${tmp_response}" >&2
    rm -f "${tmp_response}"
    continue
  fi
  if [[ ${http_status} -eq 422 ]]; then
    echo "Dependabot did not queue an update for ${ecosystem} (status 422)." >&2
    cat "${tmp_response}" >&2
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
      {
        echo "### ${ecosystem}"
        echo "Dependabot did not queue an update (status 422)."
        cat "${tmp_response}"
        echo
      } >> "${GITHUB_STEP_SUMMARY}"
    fi
    rm -f "${tmp_response}"
    continue
  fi
  if [[ ${http_status} -ge 400 ]]; then
    echo "Failed to trigger Dependabot for ${ecosystem} (status ${http_status})" >&2
    cat "${tmp_response}" >&2
    rm -f "${tmp_response}"
    exit 1
  fi
  rm -f "${tmp_response}"
done
