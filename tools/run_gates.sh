#!/bin/sh
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)
build="$root/build/cmake"
target="$root/build/harness-target"

cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$build"
RSC_LIB_DIR="$build" cargo build --release --locked \
	--manifest-path "$root/harness/Cargo.toml" --target-dir "$target"

status=0
"$build/rsc_libm_check" --quick || status=1
"$target/release/rsv3_harness" gate "$@" || status=1
if [ "$status" -eq 0 ]; then
	echo "ALL GATES PASS"
else
	echo "GATES FAIL"
fi
exit "$status"
