"""What the GPU sort costs, against a comparison sort on the host.

Every timing restores the working buffer from a pristine device buffer before
sorting, both inside the timed region, because a sort run twice on the same
buffer measures the second run on sorted input. That restore is a
device-to-device copy and is reported as a floor rather than subtracted out.

The host baseline is the stdlib's `sort` on one core. That is not a like-for-
like comparison and is not meant to be: it is the thing you would otherwise
have written. `mm_radix_sort` on the CPU is the interesting comparison, and
the README carries it.

Unified memory means there is no host-to-device transfer to account for here.
On a discrete GPU that transfer is usually what decides whether a sort is
worth shipping off the CPU at all.

Times are nanoseconds per element. Lower is better.
"""

from max.gpu.host import DeviceContext
from mm_radix_sort_gpu import gpu_radix_sort
from std.random import random_float64, random_ui64, seed
from std.sys.info import bit_width_of
from std.time import perf_counter_ns

comptime _SIZES = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24]
comptime _REPEATS = 20


def fixed(value: Float64, decimals: Int = 3) -> String:
    """Formats `value` with exactly `decimals` digits after the point."""
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = Int(value * Float64(scale) + 0.5)
    var digits = String(scaled % scale)
    while digits.byte_length() < decimals:
        digits = String("0", digits)
    return String(scaled // scale, ".", digits)


def rjust(text: String, width: Int) -> String:
    """Right-aligns `text` in a field `width` wide."""
    var out = text.copy()
    while out.byte_length() < width:
        out = String(" ", out)
    return out


def ljust(text: String, width: Int) -> String:
    """Left-aligns `text` in a field `width` wide."""
    var out = text.copy()
    while out.byte_length() < width:
        out += " "
    return out


def bench[D: DType](ctx: DeviceContext, count: Int) raises:
    """Times one type at one size and prints a row."""
    seed(1)
    var host_values = List[Scalar[D]](unsafe_uninit_length=count)
    comptime if D.is_floating_point():
        for i in range(count):
            host_values[i] = Scalar[D](random_float64() * 2000.0 - 1000.0)
    else:
        # Integer keys must span the full width of their type. An earlier
        # version drew every key from `[0, 4e9)`, which leaves the top half of
        # a `UInt64` zero -- and `mm_radix_sort` skips a pass whose digit never
        # varies, so the CPU column was doing three passes against the GPU's
        # sixteen. The GPU sort has no such shortcut, so the comparison
        # measured two different amounts of work.
        comptime W = bit_width_of[D]()
        comptime if W >= 64:
            for i in range(count):
                host_values[i] = rebind[Scalar[D]](random_ui64(0, UInt64.MAX))
        else:
            comptime LIMIT = (UInt64(1) << UInt64(W)) - 1
            for i in range(count):
                host_values[i] = Scalar[D](Int(random_ui64(0, LIMIT)))

    var pristine = ctx.enqueue_create_buffer[D](count)
    var working = ctx.enqueue_create_buffer[D](count)
    with pristine.map_to_host() as host:
        for i in range(count):
            host[i] = host_values[i]
    ctx.synchronize()

    # The copy floor: what a run costs before any sorting happens.
    var floor = Float64(1e30)
    for _ in range(_REPEATS):
        var start = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=working, src_buf=pristine)
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < floor:
            floor = elapsed

    var best = Float64(1e30)
    for _ in range(_REPEATS):
        var start = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=working, src_buf=pristine)
        gpu_radix_sort(ctx, working, count)
        ctx.synchronize()
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < best:
            best = elapsed

    # Sorted is not enough -- a kernel that wrote a constant would pass that.
    var expected = host_values.copy()
    sort(expected)
    var ok = True
    with working.map_to_host() as host:
        for i in range(count):
            if host[i] != expected[i]:
                ok = False
                break
    if not ok:
        raise Error("gpu_radix_sort disagreed with sort at n=", count)

    var host_best = Float64(1e30)
    for _ in range(3):
        var scratch = host_values.copy()
        var start = perf_counter_ns()
        sort(scratch)
        var elapsed = Float64(perf_counter_ns() - start)
        if elapsed < host_best:
            host_best = elapsed

    var n = Float64(count)
    print(
        ljust(String(D), 9),
        rjust(String(count), 10),
        rjust(fixed(floor / n), 8),
        rjust(fixed(best / n), 9),
        rjust(fixed(host_best / n, 2), 10),
        rjust(String(fixed(host_best / best, 1), "x"), 9),
    )


def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name())
    print("Nanoseconds per element, copy-restore included. Lower is better.\n")
    print(
        ljust("type", 9),
        rjust("n", 10),
        rjust("copy", 8),
        rjust("gpu", 9),
        rjust("host sort", 10),
        rjust("vs host", 9),
    )
    comptime dtypes = [DType.uint32, DType.float32, DType.uint64]
    comptime for d in range(len(dtypes)):
        comptime dtype = dtypes[d]
        comptime for i in range(len(_SIZES)):
            comptime size = _SIZES[i]
            bench[dtype](ctx, size)
