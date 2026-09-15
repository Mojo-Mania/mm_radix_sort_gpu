"""Tests for the GPU radix sort.

The oracle is the stdlib's `sort`, run on the host over the same values: the
GPU sort is correct exactly when it produces the sequence a comparison sort
produces. Every test below is that comparison.
"""

from max.gpu.host import DeviceContext
from mm_radix_sort_gpu import TILE, gpu_radix_sort
from std.random import random_float64, random_ui64, seed
from std.testing import TestSuite, assert_equal, assert_true


def _assert_matches_sort[
    D: DType, //
](ctx: DeviceContext, values: List[Scalar[D]], context: String) raises:
    """Sorts `values` on the device and compares against `sort` on the host."""
    var count = len(values)
    var expected = values.copy()
    sort(expected)

    var buffer = ctx.enqueue_create_buffer[D](count if count > 0 else 1)
    if count > 0:
        with buffer.map_to_host() as host:
            for i in range(count):
                host[i] = values[i]
    gpu_radix_sort(ctx, buffer, count)
    ctx.synchronize()

    if count == 0:
        return
    # The comparison is written out rather than handed to `assert_equal` per
    # element: the message argument is built eagerly, so a million passing
    # elements meant a million `String` constructions, which took longer than
    # the sorts did.
    with buffer.map_to_host() as host:
        for i in range(count):
            if host[i] != expected[i]:
                assert_equal(
                    host[i],
                    expected[i],
                    String(context, ": element ", i, " of ", count),
                )


def _random[D: DType](ctx: DeviceContext, count: Int, label: String) raises:
    var values = List[Scalar[D]](unsafe_uninit_length=count)
    comptime if D.is_floating_point():
        for i in range(count):
            values[i] = Scalar[D](random_float64() * 2000.0 - 1000.0)
    else:
        for i in range(count):
            var raw = Int(random_ui64(0, 60000))
            values[i] = Scalar[D](raw - 30000 if D.is_signed() else raw)
    _assert_matches_sort(ctx, values, label)


def test_degenerate_lengths() raises:
    var ctx = DeviceContext()
    _assert_matches_sort(ctx, List[UInt32](), "empty")
    _assert_matches_sort(ctx, [UInt32(42)], "one element")
    _assert_matches_sort(ctx, [UInt32(2), 1], "two, reversed")
    _assert_matches_sort(ctx, [UInt32(1), 2], "two, ordered")


def test_all_equal() raises:
    var ctx = DeviceContext()
    var values = List[UInt32](unsafe_uninit_length=50000)
    for i in range(50000):
        values[i] = 7
    _assert_matches_sort(ctx, values, "fifty thousand equal values")


def test_already_sorted_and_reversed() raises:
    var ctx = DeviceContext()
    var ascending = List[Int32](unsafe_uninit_length=50000)
    var descending = List[Int32](unsafe_uninit_length=50000)
    for i in range(50000):
        ascending[i] = Int32(i - 25000)
        descending[i] = Int32(25000 - i)
    _assert_matches_sort(ctx, ascending, "already ascending")
    _assert_matches_sort(ctx, descending, "descending")


def test_extremes() raises:
    var ctx = DeviceContext()
    _assert_matches_sort(
        ctx, [Int8.MIN, Int8.MAX, 0, -1, 1, Int8.MIN, Int8.MAX], "int8 extremes"
    )
    _assert_matches_sort(
        ctx, [UInt64.MIN, UInt64.MAX, 1, UInt64.MAX - 1], "uint64 extremes"
    )


def test_tile_boundaries() raises:
    """Sizes either side of a tile, where the partial-tile guards live."""
    var ctx = DeviceContext()
    seed(3)
    for count in [
        1,
        2,
        TILE - 1,
        TILE,
        TILE + 1,
        2 * TILE - 1,
        2 * TILE,
        2 * TILE + 1,
    ]:
        _random[DType.uint32](ctx, count, String("uint32 n=", count))


def test_every_width_and_sign() raises:
    var ctx = DeviceContext()
    seed(5)
    comptime dtypes = [
        DType.uint8,
        DType.int8,
        DType.uint16,
        DType.int16,
        DType.uint32,
        DType.int32,
        DType.uint64,
        DType.int64,
    ]
    comptime for d in range(len(dtypes)):
        comptime dtype = dtypes[d]
        for count in [3, 1000, 20000]:
            _random[dtype](ctx, count, String(dtype, " n=", count))


def test_floats() raises:
    var ctx = DeviceContext()
    seed(6)
    comptime dtypes = [DType.float32, DType.float64]
    comptime for d in range(len(dtypes)):
        comptime dtype = dtypes[d]
        for count in [3, 1000, 20000]:
            _random[dtype](ctx, count, String(dtype, " n=", count))


def test_float_special_values() raises:
    """Zeroes, denormals and infinities, but never NaN -- it has no order."""
    var ctx = DeviceContext()
    var values = [
        Float64(0.0),
        -0.0,
        1.0,
        -1.0,
        Float64.MAX,
        Float64.MIN,
        5e-324,
        -5e-324,
        1e308,
        -1e308,
    ]
    var buffer = ctx.enqueue_create_buffer[DType.float64](len(values))
    with buffer.map_to_host() as host:
        for i in range(len(values)):
            host[i] = values[i]
    gpu_radix_sort(ctx, buffer, len(values))
    ctx.synchronize()
    with buffer.map_to_host() as host:
        for i in range(1, len(values)):
            assert_true(
                host[i - 1] <= host[i],
                String("float specials out of order at ", i),
            )


def test_large() raises:
    """Past the point where a single scan block walks many tiles."""
    var ctx = DeviceContext()
    seed(7)
    _random[DType.uint32](ctx, 1 << 20, "uint32 n=1Mi")
    _random[DType.float32](ctx, 1 << 20, "float32 n=1Mi")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
