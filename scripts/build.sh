#!/bin/bash
# build.sh <rom>... [--keys aes_keys.txt] [--out dir] [--wait] [--32bit]
#
# Builds Pocket Native and games for the handheld from your own ROMs,
# .3ds/.cci or .cia files, compressed (.zcci/.zcia) or not, with your
# console's aes_keys.txt where they are encrypted, inside a container
# (podman or docker). With no ROM it asks for them, and
# --wait waits for Enter at the end. --32bit builds each game's code for a
# 32-bit Android device instead, for the 32-bit app to play. Leaves
# "Pocket Native" in the output directory, build/ by default, to copy into
# roms/ports on the card.
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
roms=() keys="" out="$repo/build" bits=64
while [ $# -gt 0 ]; do
    case "$1" in
        --keys) keys="$(realpath "$2")"; shift 2 ;;
        --out) out="$(realpath -m "$2")"; shift 2 ;;
        --wait) trap 'code=$?; read -rp "press Enter to close " _ || true; exit $code' EXIT; shift ;;
        --32bit) bits=32; shift ;;
        -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
        *) roms+=("$(realpath -m "$1")"); shift ;;
    esac
done

# a file picker from the desktop's own toolkit, one path a line
pick() {
    local title="$1" name="$2" patterns="$3" many="${4:-}"
    if command -v kdialog >/dev/null && { [ "${XDG_CURRENT_DESKTOP:-}" = KDE ] || ! command -v zenity >/dev/null; }; then
        kdialog --title "$title" --getopenfilename ${many:+--multiple --separate-output} "$HOME" "$patterns|$name" 2>/dev/null
    elif command -v zenity >/dev/null; then
        zenity --file-selection --title="$title" ${many:+--multiple} --file-filter="$name | $patterns" 2>/dev/null | tr '|' '\n'
    fi
}

if [ ${#roms[@]} -eq 0 ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    mapfile -t roms < <(pick "Pick the games to build" "3DS games" "*.3ds *.cci *.cia *.zcci *.zcia *.3DS *.CCI *.CIA *.ZCCI *.ZCIA" many || true)
    [ ${#roms[@]} -eq 0 ] && { echo "no game picked"; exit 2; }
fi
if [ ${#roms[@]} -eq 0 ]; then
    echo "usage: scripts/build.sh <rom>... [--keys aes_keys.txt] [--out dir] [--32bit]"
    exit 2
fi
for rom in "${roms[@]}"; do
    [ -f "$rom" ] || { echo "no such file: $rom"; exit 2; }
done
engine=$(command -v podman || command -v docker) || { echo "the build needs podman or docker"; exit 1; }
image=pocket-native-build:$(sha256sum "$repo/scripts/container/Containerfile" | cut -c1-12)
if ! "$engine" image inspect "$image" >/dev/null 2>&1; then
    echo "== making the build container, once"
    "$engine" build -f "$repo/scripts/container/Containerfile" -t "$image" "$repo/scripts/container"
fi

mkdir -p "$out/.work"
user=(--user "$(id -u):$(id -g)")
[ "$(basename "$engine")" = podman ] && user=(--userns=keep-id)
failed=()
for rom in "${roms[@]}"; do
    mounts=(-v "$repo:/src:ro,Z" -v "$out/.work:/work" -v "$out:/out" -v "$(dirname "$rom"):/rom:ro,Z")
    args=("/rom/$(basename "$rom")")
    # a game takes the keys given, else the aes_keys.txt beside it or in the
    # kit, for an encrypted one, and a .cia else the one picked when asked
    rom_keys="$keys"
    if [ -z "$rom_keys" ]; then
        for near in "$(dirname "$rom")/aes_keys.txt" "$repo/aes_keys.txt"; do
            [ -f "$near" ] && { rom_keys="$near"; break; }
        done
        if [ -z "$rom_keys" ] && [[ "${rom,,}" == *.cia || "${rom,,}" == *.zcia ]] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
            keys=$(pick "A .cia needs your console's aes_keys.txt, pick it" "aes_keys.txt" "aes_keys.txt *.txt" || true)
            rom_keys="$keys"
        fi
    fi
    if [ -n "$rom_keys" ]; then
        mounts+=(-v "$(dirname "$rom_keys"):/keys:ro")
        args+=("/keys/$(basename "$rom_keys")")
    fi
    "$engine" run --rm "${user[@]}" "${mounts[@]}" -e CARGO_HOME=/work/cargo -e CARGO_TARGET_DIR=/work/target -e HOME=/work -e BITS="$bits" \
        "$image" bash /src/scripts/container/inside.sh "${args[@]}" || failed+=("$rom")
done
bash "$repo/scripts/update.sh" --check || true
if [ ${#failed[@]} -gt 0 ]; then
    printf '== these did not build:\n'
    printf '   %s\n' "${failed[@]}"
    exit 1
fi
