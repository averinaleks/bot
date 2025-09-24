#!/usr/bin/env bash
set -euo pipefail

PATCH_SPECS="${1:-/tmp/security/linux-pam-CVE-2024-10963.patch}"
CHANGELOG_HELPER="${2:-/tmp/security/update_pam_changelog.py}"
DIST_CODENAME="${3:-noble}"
BUILD_ROOT="${4:-/tmp/security/pam-build}"
OUTPUT_DIR="${5:-/tmp/pam-fixed}"

read -r -a PATCH_PATHS <<<"${PATCH_SPECS}"

determine_sentinels() {
  local patch_name
  patch_name=$(basename "$1")
  case "${patch_name}" in
    linux-pam-CVE-2024-10963.patch)
      printf '%s\n' \
        'modules/pam_access/pam_access.c:nodns'
      ;;
    linux-pam-CVE-2024-10041.patch)
      printf '%s\n' \
        'modules/pam_unix/passverify.c:The helper has to be invoked to deal with' \
        'modules/pam_unix/pam_unix_acct.c:pam_syslog(pamh, euid == 0 ? LOG_ERR : LOG_DEBUG)'
      ;;
    *)
      return 1
      ;;
  esac
}

apply_patch_or_skip() {
  local patch_file="$1"
  local -a sentinels

  mapfile -t sentinels < <(determine_sentinels "${patch_file}" || true)

  if patch -p1 --forward <"${patch_file}"; then
    return 0
  fi

  if [[ ${#sentinels[@]} -eq 0 ]]; then
    echo "Не удалось применить патч ${patch_file}" >&2
    return 1
  fi

  local all_present=1
  for entry in "${sentinels[@]}"; do
    local target=${entry%%:*}
    local marker=${entry#*:}
    if [[ ! -f "${target}" ]] || ! grep -Fq "${marker}" "${target}"; then
      all_present=0
      break
    fi
  done

  if [[ ${all_present} -eq 1 ]]; then
    echo "Патч ${patch_file} уже присутствует, пропускаем." >&2
    return 0
  fi

  echo "Не удалось применить патч ${patch_file}" >&2
  return 1
}

mkdir -p "${BUILD_ROOT}"
cd "${BUILD_ROOT}"

SRC_LIST="/etc/apt/sources.list.d/${DIST_CODENAME}-src.list"
cat <<SRC >"${SRC_LIST}"
deb-src http://archive.ubuntu.com/ubuntu ${DIST_CODENAME} main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu ${DIST_CODENAME}-updates main restricted universe multiverse
deb-src http://security.ubuntu.com/ubuntu ${DIST_CODENAME}-security main restricted universe multiverse
SRC

apt-get update
apt-get install -y --no-install-recommends devscripts equivs
apt-get build-dep -y pam
apt-get source -y pam

pam_src_dir=$(find . -maxdepth 1 -type d -name 'pam-*' | head -n 1)
if [[ -z "${pam_src_dir}" ]]; then
  echo "Не удалось найти исходники PAM" >&2
  exit 1
fi

rm -rf pam-src
mv "${pam_src_dir}" pam-src

pushd pam-src >/dev/null
for patch_file in "${PATCH_PATHS[@]}"; do
  [[ -z "${patch_file}" ]] && continue
  if [[ ! -f "${patch_file}" ]]; then
    echo "Файл патча ${patch_file} не найден" >&2
    exit 1
  fi
  apply_patch_or_skip "${patch_file}"
done
python3 "${CHANGELOG_HELPER}"
export DEB_BUILD_OPTIONS=nocheck
dpkg-buildpackage -b -uc -us
popd >/dev/null

mkdir -p "${OUTPUT_DIR}"
cp libpam-modules_* libpam-modules-bin_* libpam-runtime_* libpam0g_* "${OUTPUT_DIR}/"
dpkg -i "${OUTPUT_DIR}"/*.deb

rm -rf pam-src "${SRC_LIST}"
apt-get purge -y --auto-remove devscripts equivs
apt-get clean
rm -rf /var/lib/apt/lists/*
