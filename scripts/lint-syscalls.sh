#!/usr/bin/env bash
# a raw syscall that Android's seccomp filter blocks would kill the game
# process with SIGSYS instead of returning an error, so raw syscalls are
# only allowed behind cfg(not(target_os = "android")), apart from the
# allowlist below: gettid, which bionic itself uses and Android allows.
set -euo pipefail
cd "$(dirname "$0")/.."

violations="$(grep -rn "libc::syscall\|SYS_[a-z]" frontend/src engine --include="*.rs" \
    | grep -v "frontend/src/readahead.rs" \
    | grep -v "frontend/src/profiler.rs" || true)"
if [ -n "$violations" ]; then
    echo "raw syscalls outside the allowlist (readahead.rs, profiler.rs):" >&2
    echo "$violations" >&2
    exit 1
fi

# ioprio_set is blocked by Android's seccomp filter, so it must be
# compiled out there; the call may only live in readahead.rs, guarded.
if grep -q "ioprio_set" frontend/src/readahead.rs \
    && ! grep -q 'cfg(not(target_os = "android"))' frontend/src/readahead.rs; then
    echo "frontend/src/readahead.rs calls ioprio_set without excluding Android" >&2
    exit 1
fi
