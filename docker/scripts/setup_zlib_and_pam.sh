#!/usr/bin/env bash
set -euo pipefail

# Ensure we are operating in the build directory where zlib archive resides.
cd /tmp/build

# List potential symlinks that might interfere with extraction. Ignoring failures keeps the build resilient.
find /usr -type l -lname '*..*' -print 2>/dev/null || true

SAFE_TAR_DIR="$(mktemp -d)"
trap 'rm -rf "$SAFE_TAR_DIR"' EXIT

# Extract zlib sources into a clean directory, mitigating CVE-2025-45582 by avoiding reused directories.
tar --keep-directory-symlink --no-overwrite-dir -xf zlib.tar.gz -C "$SAFE_TAR_DIR"
rm -rf zlib-src
mv "$SAFE_TAR_DIR"/"zlib-${ZLIB_VERSION}" zlib-src

# Prepare PAM sources with the required repositories enabled.
rm -rf /etc/apt/sources.list.d/noble-src.list
cat <<'SRC' >/etc/apt/sources.list.d/noble-src.list
deb-src http://archive.ubuntu.com/ubuntu noble main restricted universe multiverse
deb-src http://archive.ubuntu.com/ubuntu noble-updates main restricted universe multiverse
deb-src http://security.ubuntu.com/ubuntu noble-security main restricted universe multiverse
SRC

apt-get update
apt-get build-dep -y pam
apt-get source -y pam
pam_src_dir=$(find . -maxdepth 1 -type d -name 'pam-*' | head -n 1)
rm -rf pam-src
mv "$pam_src_dir" pam-src
