"""Sorting on the GPU, end to end."""

from max.gpu.host import DeviceContext
from mm_radix_sort_gpu import gpu_radix_sort
from std.random import random_float64, random_ui64, seed


def integers(ctx: DeviceContext) raises:
    print("--- signed integers, on the device ---")
    var values = [Int32(5), -3, 9, -1, 0, -128, 127]
    var keys = ctx.enqueue_create_buffer[DType.int32](len(values))
    with keys.map_to_host() as host:
        for i in range(len(values)):
            host[i] = values[i]

    gpu_radix_sort(ctx, keys, len(values))
    ctx.synchronize()

    with keys.map_to_host() as host:
        var out = List[Int32]()
        for i in range(len(values)):
            out.append(host[i])
        print(out)


def floats(ctx: DeviceContext) raises:
    print("\n--- floats, including negatives ---")
    # Metal has no `double`, so a kernel cannot load a Float64 at all. The
    # sort moves the raw bits instead of the values, which costs nothing --
    # ordering is all a sort ever needs -- and makes Float64 work anyway.
    var values = [Float64(2.5), -1.0, 0.0, -7.25, 1e300, -1e300]
    var keys = ctx.enqueue_create_buffer[DType.float64](len(values))
    with keys.map_to_host() as host:
        for i in range(len(values)):
            host[i] = values[i]

    gpu_radix_sort(ctx, keys, len(values))
    ctx.synchronize()

    with keys.map_to_host() as host:
        var out = List[Float64]()
        for i in range(len(values)):
            out.append(host[i])
        print(out)


def at_scale(ctx: DeviceContext) raises:
    print("\n--- a million keys, checked against the stdlib ---")
    var count = 1 << 20
    seed(1)
    var expected = List[UInt32](unsafe_uninit_length=count)
    var keys = ctx.enqueue_create_buffer[DType.uint32](count)
    with keys.map_to_host() as host:
        for i in range(count):
            var value = UInt32(Int(random_ui64(0, 4_000_000_000)))
            host[i] = value
            expected[i] = value
    sort(expected)

    gpu_radix_sort(ctx, keys, count)
    ctx.synchronize()

    with keys.map_to_host() as host:
        for i in range(count):
            if host[i] != expected[i]:
                raise Error("the GPU disagreed with sort at ", i)
    print(count, "keys agree with the stdlib exactly")


def main() raises:
    var ctx = DeviceContext()
    print("device:", ctx.name(), "\n")
    integers(ctx)
    floats(ctx)
    at_scale(ctx)
