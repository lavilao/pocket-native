#!/bin/bash
# inside.sh <out.apk> <keystore> [package] [abi]: the Android app, built in
# the container: /src the repository, read only, /work the scratch space kept
# between builds. abi is arm64-v8a, the default, or armeabi-v7a for a 32-bit
# device, whose games have to be built without RECOMP_FASTMEM, which
# 3dsrecomp does unless RECOMP_CFLAGS says otherwise.
set -euo pipefail
out="$1" keystore="$2" package="${3:-io.github.pocketnative}" abi="${4:-arm64-v8a}"
case "$abi" in
    arm64-v8a)
        target=aarch64-linux-android
        export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$NDK_BIN/aarch64-linux-android30-clang"
        export CC_aarch64_linux_android="$NDK_BIN/aarch64-linux-android30-clang"
        export AR_aarch64_linux_android="$NDK_BIN/llvm-ar"
        ;;
    armeabi-v7a)
        target=armv7-linux-androideabi
        export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$NDK_BIN/armv7a-linux-androideabi29-clang"
        export CC_armv7_linux_androideabi="$NDK_BIN/armv7a-linux-androideabi29-clang"
        export AR_armv7_linux_androideabi="$NDK_BIN/llvm-ar"
        ;;
    *) echo "unknown abi $abi"; exit 2 ;;
esac
cd /src
cargo build --release --locked --target "$target" -p pocket-native-android
"$NDK_BIN/llvm-strip" -o /work/libpocketnative.so "$CARGO_TARGET_DIR/$target/release/libpocketnative.so"
# the profiles, without the overrides beside some of them
rm -rf /work/assets && mkdir -p /work/assets/titles
for profile in titles/*/profile.toml; do
    mkdir -p "/work/assets/$(dirname "$profile")"
    cp "$profile" "/work/assets/$profile"
done
# the licenses, as the kit puts them beside the program
python3 tools/third_party.py /work/assets/THIRD_PARTY_LICENSES.txt pocket-native-android "$target" >/dev/null
cp LICENSE CREDITS.md assets/fonts/OFL.txt /work/assets/
label="Pocket Native"
[ "$package" = io.github.pocketnative ] || label="Pocket Native (test)"
bash scripts/android/make_apk.sh "$package" "$label" /work/libpocketnative.so /work/assets "$out" "$keystore" "$abi"
