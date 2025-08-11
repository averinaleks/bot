#!/usr/bin/env sh
# Check for CVE-2020-14356: null pointer dereference in cgroupv2 on reboot
# The issue affects kernels before 5.7.10 and certain Ubuntu 4.15 kernels.
set -e

KERNEL_RELEASE="$(uname -r)"
KERNEL_VERSION="$(printf '%s' "$KERNEL_RELEASE" | cut -d'-' -f1)"

is_less_than() {
    # returns true if version $1 < version $2
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" = "$1" ] && [ "$1" != "$2" ]
}

if is_less_than "$KERNEL_VERSION" "5.7.10"; then
    echo "Kernel $KERNEL_RELEASE may be vulnerable to CVE-2020-14356 (needs >=5.7.10)." >&2
    exit 1
fi

# Handle Ubuntu 4.15 kernels specifically
if echo "$KERNEL_RELEASE" | grep -q '^4\.15\.0-'; then
    PATCH="$(printf '%s' "$KERNEL_RELEASE" | cut -d'-' -f2 | cut -d'.' -f1)"
    if [ "$PATCH" -lt 118 ]; then
        echo "Kernel $KERNEL_RELEASE is vulnerable to CVE-2020-14356 (needs >=4.15.0-118)." >&2
        exit 1
    fi
fi

echo "Kernel $KERNEL_RELEASE is not vulnerable to CVE-2020-14356." 
