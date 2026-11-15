#!/bin/sh
set -eu

root=$(cd "$(dirname "$0")/.." && pwd)

modes=default
if [ "${1:-}" = "--mode" ]; then
	modes=$2
	shift 2
fi
if [ "$modes" = "all" ]; then
	modes="default native-math gjk native-math-gjk"
fi

gate_mode() {
	mode=$1
	shift
	case "$mode" in
	default)
		suffix=""
		features=""
		native_math=OFF
		box_triangle_gjk=OFF
		;;
	native-math)
		suffix="-native-math"
		features="native-math"
		native_math=ON
		box_triangle_gjk=OFF
		;;
	gjk)
		suffix="-gjk"
		features="box-triangle-gjk"
		native_math=OFF
		box_triangle_gjk=ON
		;;
	native-math-gjk)
		suffix="-native-math-gjk"
		features="native-math,box-triangle-gjk"
		native_math=ON
		box_triangle_gjk=ON
		;;
	*)
		echo "unknown mode: $mode (expected default, native-math, gjk, native-math-gjk or all)" >&2
		exit 2
		;;
	esac
	build="$root/build/cmake$suffix"
	target="$root/build/harness-target$suffix"

	cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
		-DRSC_NATIVE_MATH="$native_math" \
		-DRSC_BOX_TRIANGLE_GJK="$box_triangle_gjk"
	cmake --build "$build"
	RSC_LIB_DIR="$build" cargo build --release --locked \
		--manifest-path "$root/harness/Cargo.toml" --target-dir "$target" \
		--features "$features"

	mode_status=0
	"$build/rsc_libm_check" --quick || mode_status=1
	"$build/rsc_lanes_check" --quick || mode_status=1
	"$target/release/rsv3_harness" gate "$@" || mode_status=1
	return "$mode_status"
}

status=0
for mode in $modes; do
	echo "MODE $mode"
	gate_mode "$mode" "$@" || status=1
done
if [ "$status" -eq 0 ]; then
	echo "ALL GATES PASS"
else
	echo "GATES FAIL"
fi
exit "$status"
