"""`gpu_radix_sort` -- an LSD radix sort that runs on the GPU.

The host side is a loop over digits. Each pass launches three kernels, and the
buffers ping-pong, so after an even number of passes the answer is back where
it started and after an odd number it is in the scratch buffer. The driver
copies it back when that happens rather than making the caller care.

```mojo
from max.gpu.host import DeviceContext
from mm_radix_sort_gpu import gpu_radix_sort

var ctx = DeviceContext()
var keys = ctx.enqueue_create_buffer[DType.uint32](1 << 20)
# ... fill ...
gpu_radix_sort(ctx, keys, 1 << 20)
ctx.synchronize()
```

Signed integers and floats sort correctly, negatives included, through the
same order-preserving mapping the CPU package uses -- see `_bits`.
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from std.sys.info import bit_width_of

from ._bits import pass_count
from .kernels import (
    BITS,
    BUCKETS,
    TILE,
    THREADS,
    histogram_kernel,
    scan_kernel,
    scatter_kernel,
)


def gpu_radix_sort[
    D: DType, //
](ctx: DeviceContext, keys: DeviceBuffer[D], count: Int,) raises:
    """Sorts the first `count` elements of `keys` in ascending order.

    The sort is stable and runs entirely on the device. It allocates two
    scratch buffers -- one the length of the keys, one holding
    `(count / TILE + 1) * BUCKETS` counters twice over -- and makes
    `ceil(bit_width_of[D]() / 4)` passes over the data.

    Work is enqueued, not awaited. Call `ctx.synchronize()` before reading the
    result.

    Parameters:
        D: The element type. Signed integers and floats sort correctly.

    Args:
        ctx: The device to run on.
        keys: The buffer to sort, in place.
        count: How many elements to sort.

    Raises:
        Error: If a device allocation or launch fails.
    """
    if count < 2:
        return

    comptime PASSES = pass_count[D, BITS]()
    var blocks = (count + TILE - 1) // TILE
    var counters = blocks * BUCKETS

    var scratch = ctx.enqueue_create_buffer[D](count)
    var block_counts = ctx.enqueue_create_buffer[DType.uint32](counters)
    var block_offsets = ctx.enqueue_create_buffer[DType.uint32](counters)

    var flipped = False
    comptime for p in range(PASSES):
        var source = scratch if flipped else keys
        var target = keys if flipped else scratch

        ctx.enqueue_function[histogram_kernel[D]](
            source.unsafe_ptr(),
            block_counts.unsafe_ptr(),
            Int32(count),
            Int32(p),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        ctx.enqueue_function[scan_kernel](
            block_counts.unsafe_ptr(),
            block_offsets.unsafe_ptr(),
            Int32(blocks),
            grid_dim=1,
            block_dim=THREADS,
        )
        ctx.enqueue_function[scatter_kernel[D]](
            source.unsafe_ptr(),
            target.unsafe_ptr(),
            block_offsets.unsafe_ptr(),
            Int32(count),
            Int32(p),
            grid_dim=blocks,
            block_dim=THREADS,
        )
        flipped = not flipped

    # An odd number of passes leaves the answer in the scratch buffer.
    comptime if PASSES % 2 == 1:
        ctx.enqueue_copy(dst_buf=keys, src_buf=scratch)
