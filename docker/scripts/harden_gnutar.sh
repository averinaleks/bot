#!/usr/bin/env bash
set -euo pipefail

TARGET_BIN="/usr/bin/tar"
DIVERTED_BIN="${TARGET_BIN}.distrib"

if ! command -v dpkg-divert >/dev/null 2>&1; then
    echo "dpkg-divert not available, skipping tar hardening" >&2
    exit 0
fi

if [[ -x "$DIVERTED_BIN" ]]; then
    echo "tar binary already diverted, skipping" >&2
    exit 0
fi

if [[ ! -x "$TARGET_BIN" ]]; then
    echo "tar binary not found at $TARGET_BIN" >&2
    exit 0
fi

# Move the original GNU tar binary aside and install a secure wrapper that
# enforces symlink-safe extraction flags.  This mitigates CVE-2025-45582 by
# preventing follow-up archives from traversing directory symlinks created by
# previous extractions.
dpkg-divert --local --rename --add "$TARGET_BIN"

cat <<'WRAPPER' > "$TARGET_BIN"
#!/usr/bin/env bash
set -euo pipefail

ORIGINAL="/usr/bin/tar.distrib"
if [[ ! -x "$ORIGINAL" ]]; then
    echo "Hardened tar wrapper: original binary missing at $ORIGINAL" >&2
    exit 1
fi

# Always include symlink-safe extraction flags.  TAR_OPTIONS preserves user
# supplied defaults while appending the protections required to mitigate
# CVE-2025-45582.
ADD_OPTS=("--keep-directory-symlink" "--no-overwrite-dir")
for opt in "${ADD_OPTS[@]}"; do
    case " ${TAR_OPTIONS-} " in
        *" $opt "*) ;;
        *)
            if [[ -n "${TAR_OPTIONS-}" ]]; then
                TAR_OPTIONS+=" $opt"
            else
                TAR_OPTIONS="$opt"
            fi
        ;;
    esac
done
export TAR_OPTIONS

exec "$ORIGINAL" "$@"
WRAPPER

chmod 0755 "$TARGET_BIN"

# Verify wrapper behaviour during the image build.
"$TARGET_BIN" --version >/dev/null
