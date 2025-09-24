#!/usr/bin/env bash
set -euo pipefail

PATCH_PATH="${1:-/tmp/security/linux-pam-CVE-2024-10963.patch}"
CHANGELOG_HELPER="${2:-/tmp/security/update_pam_changelog.py}"
DIST_CODENAME="${3:-noble}"
BUILD_ROOT="${4:-/tmp/security/pam-build}"
OUTPUT_DIR="${5:-/tmp/pam-fixed}"

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
patch -p1 < "${PATCH_PATH}"
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
