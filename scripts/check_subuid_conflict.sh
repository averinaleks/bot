#!/usr/bin/env sh
# Check for CVE-2024-56433 by ensuring /etc/subuid range does not overlap
# with existing user UIDs.
set -e
USER_NAME="${1:-$(id -un)}"
SUBUID_FILE="/etc/subuid"

if [ ! -f "$SUBUID_FILE" ]; then
    echo "\"$SUBUID_FILE\" not found" >&2
    exit 1
fi

# Highest UID from /etc/passwd (non-system)
MAX_UID=$(awk -F: '$3 >= 1000 { if($3 > max) max=$3 } END { print max }' /etc/passwd)

RANGE=$(grep "^$USER_NAME:" "$SUBUID_FILE" | cut -d: -f2-)
if [ -z "$RANGE" ]; then
    echo "No subuid range configured for $USER_NAME" >&2
    exit 1
fi

START=${RANGE%:*}
COUNT=${RANGE##*:}
END=$((START + COUNT - 1))

if [ "$START" -le "$MAX_UID" ]; then
    echo "Warning: subuid range $START-$END overlaps with existing UIDs (max UID $MAX_UID)." >&2
    echo "Consider assigning a range starting above $((MAX_UID + 100000))." >&2
    exit 1
fi

printf 'subuid range %s-%s is safe (max UID %s)\n' "$START" "$END" "$MAX_UID"
