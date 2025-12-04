#!/usr/bin/env bash
set -euo pipefail

PATCH_SPECS="${1:-/tmp/security/linux-pam-hardening.patch /tmp/security/linux-pam-doc-messaging.patch /tmp/security/linux-pam-CVE-2024-10963.patch /tmp/security/linux-pam-CVE-2024-10041.patch}"
CHANGELOG_HELPER="${2:-/tmp/security/update_pam_changelog.py}"
DIST_CODENAME="${3:-noble}"
BUILD_ROOT="${4:-/tmp/security/pam-build}"
OUTPUT_DIR="${5:-/tmp/pam-fixed}"

read -r -a PATCH_PATHS <<<"${PATCH_SPECS}"

determine_sentinels() {
  local patch_name
  patch_name=$(basename "$1")
  case "${patch_name}" in
    linux-pam-doc-messaging.patch)
      printf '%s\n' \
        'doc/adg/Makefile.am:No FO-to-PDF processor installed; skipping PDF generation.' \
        'doc/mwg/Makefile.am:No FO-to-PDF processor installed; skipping PDF generation.' \
        'doc/sag/Makefile.am:No FO-to-PDF processor installed; skipping PDF generation.'
      ;;
    linux-pam-CVE-2024-10963.patch)
      printf '%s\n' \
        'modules/pam_access/pam_access.c:nodns'
      ;;
    linux-pam-hardening.patch)
      printf '%s\n' \
        'configure.ac:AC_PROG_LEX([noyywrap])' \
        'modules/pam_extrausers/passverify.c:pam_safe_drop'
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

patch_already_present() {
  local patch_file="$1"
  local patch_name
  patch_name=$(basename "${patch_file}")

  case "${patch_name}" in
    linux-pam-doc-messaging.patch)
      python3 - "$patch_file" <<'PY'
import pathlib
import sys

targets = [
    pathlib.Path("doc/adg/Makefile.am"),
    pathlib.Path("doc/mwg/Makefile.am"),
    pathlib.Path("doc/sag/Makefile.am"),
]

expected = "No FO-to-PDF processor installed; skipping PDF generation."

try:
    contents = [path.read_text(encoding="utf-8") for path in targets]
except FileNotFoundError:
    sys.exit(1)

if all(expected in text for text in contents):
    sys.exit(0)

sys.exit(1)
PY
      return
      ;;
    linux-pam-CVE-2024-10963.patch)
      python3 - "$patch_file" <<'PY'
import pathlib
import sys

c_path = pathlib.Path("modules/pam_access/pam_access.c")
xml_path = pathlib.Path("modules/pam_access/pam_access.8.xml")

try:
    c_text = c_path.read_text(encoding="utf-8")
    xml_text = xml_path.read_text(encoding="utf-8")
except FileNotFoundError:
    sys.exit(1)

if "nodns" in c_text and "nodns" in xml_text:
    sys.exit(0)

sys.exit(1)
PY
      return
      ;;
    linux-pam-CVE-2024-10041.patch)
      python3 - "$patch_file" <<'PY'
import pathlib
import re
import sys

acct_path = pathlib.Path("modules/pam_unix/pam_unix_acct.c")
passverify_path = pathlib.Path("modules/pam_unix/passverify.c")

try:
    acct_text = acct_path.read_text(encoding="utf-8")
    pass_text = passverify_path.read_text(encoding="utf-8")
except FileNotFoundError:
    sys.exit(1)

helper_comment = "The helper has to be invoked"
acct_pattern = re.compile(r"pam_syslog\(pamh,\s*(?:\(?geteuid\(\)\s*==\s*0\)?|euid\s*==\s*0)\s*\?\s*LOG_ERR\s*:\s*LOG_DEBUG")

if helper_comment in pass_text and acct_pattern.search(acct_text):
    sys.exit(0)

sys.exit(1)
PY
      return
      ;;
    linux-pam-hardening.patch)
      python3 - "$patch_file" <<'PY'
import pathlib
import sys

cfg = pathlib.Path("configure.ac")
bigcrypt = pathlib.Path("modules/pam_extrausers/bigcrypt.c")
passverify = pathlib.Path("modules/pam_extrausers/passverify.c")
lckpwdf = pathlib.Path("modules/pam_extrausers/lckpwdf.-c")

try:
    cfg_text = cfg.read_text(encoding="utf-8")
    bigcrypt_text = bigcrypt.read_text(encoding="utf-8")
    passverify_text = passverify.read_text(encoding="utf-8")
    lckpwdf_text = lckpwdf.read_text(encoding="utf-8")
except FileNotFoundError:
    sys.exit(1)

if "AC_PROG_LEX([noyywrap])" in cfg_text and "pam_safe_drop" in passverify_text and "pam_inline.h" in bigcrypt_text and "char *create_context" in lckpwdf_text:
    sys.exit(0)

sys.exit(1)
PY
      return
      ;;
  esac

  return 1
}

apply_patch_or_skip() {
  local patch_file="$1"
  local -a sentinels

  if patch_already_present "${patch_file}"; then
    echo "Патч ${patch_file} уже присутствует (обнаружено до применения), пропускаем." >&2
    return 0
  fi

  mapfile -t sentinels < <(determine_sentinels "${patch_file}" || true)

  if patch -p1 --forward <"${patch_file}"; then
    return 0
  fi

  if patch -p1 --reverse --dry-run <"${patch_file}"; then
    echo "Патч ${patch_file} уже применён (обратное применение прошло успешно), пропускаем." >&2
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

  if patch_already_present "${patch_file}"; then
    echo "Патч ${patch_file} уже присутствует (обнаружено эвристикой), пропускаем." >&2
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
export DH_STRIP_ARGS="${DH_STRIP_ARGS:+${DH_STRIP_ARGS} }--no-dwz"
export DH_DWZ_ARGS="${DH_DWZ_ARGS:+${DH_DWZ_ARGS} }--no-dwz"
export DEB_BUILD_OPTIONS=nocheck
dpkg-buildpackage -b -uc -us
popd >/dev/null

mkdir -p "${OUTPUT_DIR}"
cp libpam-modules_* libpam-modules-bin_* libpam-runtime_* libpam0g_* "${OUTPUT_DIR}/"
apt-get install -y --no-install-recommends \
  "${OUTPUT_DIR}"/libpam0g_* \
  "${OUTPUT_DIR}"/libpam-runtime_* \
  "${OUTPUT_DIR}"/libpam-modules-bin_* \
  "${OUTPUT_DIR}"/libpam-modules_*

rm -rf pam-src "${SRC_LIST}"
apt-get purge -y --auto-remove devscripts equivs
apt-get clean
rm -rf /var/lib/apt/lists/*
