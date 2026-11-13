#!/usr/bin/env bash
# a raw syscall that Android's seccomp filter blocks would kill the game
# process with SIGSYS instead of returning an error, so raw syscalls are
# only allowed behind cfg(not(target_os = "android")), apart from the
# allowlist below: gettid, which bionic itself uses and Android allows.
set -euo pipefail
cd "$(dirname "$0")/.."

violations="$(grep -rn "libc::syscall\|SYS_[a-z]" frontend/src engine --include="*.rs" \
    | grep -v "frontend/src/readahead.rs" \
    | grep -v "frontend/src/profiler.rs" \
    | grep -v "engine/gpu/src/raster/gles/programs.rs" || true)"
if [ -n "$violations" ]; then
    echo "raw syscalls outside the allowlist (readahead.rs, profiler.rs, programs.rs gettid):" >&2
    echo "$violations" >&2
    exit 1
fi

# gettid is the one raw syscall Android's filter allows. programs.rs uses
# it from a signal handler, so it may only be that call.
if grep -n "libc::syscall\|SYS_[a-z]" engine/gpu/src/raster/gles/programs.rs \
    | grep -v SYS_gettid; then
    echo "engine/gpu/src/raster/gles/programs.rs has a raw syscall other than gettid" >&2
    exit 1
fi

# ioprio_set is blocked by Android's seccomp filter, so it must be
# compiled out there; the call may only live in readahead.rs, guarded.
if grep -q "ioprio_set" frontend/src/readahead.rs \
    && ! grep -q 'cfg(not(target_os = "android"))' frontend/src/readahead.rs; then
    echo "frontend/src/readahead.rs calls ioprio_set without excluding Android" >&2
    exit 1
fi
