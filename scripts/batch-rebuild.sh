#!/bin/bash
# batch-rebuild.sh [--out dir] [--wait] [--32bit]: builds every game already
# in the output directory, build/ by default, again with this kit, as after
# an update or with the title.hints a handheld wrote, for the handheld or,
# with --32bit, for a 32-bit Android device. --wait waits for Enter at the
# end.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
out="$repo/build" wait="" bits=64
while [ $# -gt 0 ]; do
    case "$1" in
        --out) out="$(realpath -m "$2")"; shift 2 ;;
        --wait) wait=1; shift ;;
        --32bit) bits=32; shift ;;
        -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
        *) echo "usage: scripts/batch-rebuild.sh [--out dir] [--wait] [--32bit]"; exit 2 ;;
    esac
done

# each game builds from the copy the last build left, moved out of the way
# while the build copies it back, and put back if it did not, also when a
# build stopped halfway left it there
shopt -s nullglob
games="$out/Pocket Native/pocket-native/games"
staging="$repo/.update/games"
restore() {
    for copy in "$staging"/*.3ds; do
        game="$games/$(basename "$copy" .3ds)/game.3ds"
        if [ -f "$game" ] && [ "$(stat -c %s "$game")" = "$(stat -c %s "$copy")" ]; then
            rm "$copy"
        else
            mkdir -p "$(dirname "$game")" && mv -f "$copy" "$game"
        fi
    done
}

# games moved out for the build go back however this ends
finish() {
    local code=$?
    restore
    [ -n "$wait" ] && { read -rp "press Enter to close " _ || true; }
    exit $code
}
trap finish EXIT
restore
trap 'exit 1' HUP INT TERM
built=("$games"/*/game.3ds)
if [ ${#built[@]} -eq 0 ]; then
    echo "== no games in $out to build again, build them as before"
    exit 0
fi
mkdir -p "$staging"
for game in "${built[@]}"; do
    mv "$game" "$staging/$(basename "$(dirname "$game")").3ds"
done
echo "== building ${#built[@]} game$([ ${#built[@]} = 1 ] || echo s) again"
status=0
extra=()
[ "$bits" = 32 ] && extra=(--32bit)
bash "$repo/scripts/build.sh" "$staging"/*.3ds --out "$out" "${extra[@]}" || status=$?
restore
rmdir "$staging" "$repo/.update" 2>/dev/null || true
if [ $status -eq 0 ]; then
    [ "$bits" = 32 ] \
        && echo "== done: copy \"Pocket Native\" onto the device again, over the old one" \
        || echo "== done: copy \"Pocket Native\" onto the card again, over the old one"
fi
exit $status
