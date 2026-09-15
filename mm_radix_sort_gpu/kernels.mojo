"""The three kernels one LSD radix pass is made of.

A pass over the data is: count each block's digits, work out where every
block's share of every bucket begins, then scatter. That is the same shape as
the CPU package's `lsb_radix_sort` -- histogram, prefix sum, scatter -- with
the prefix sum split in two because the blocks cannot see each other.

The digit is **4 bits wide**, not the 8 the CPU sort uses, and the reason is
the ranking in `scatter_kernel`. An LSD sort is only correct if each pass is
stable, and making a pass stable on a GPU means every element has to know its
rank among the elements of its block that share its digit. That rank comes
from a per-thread histogram held in threadgroup memory, which costs
`BUCKETS * threads` counters: 8 KiB at 4 bits, and 128 KiB at 8, which no
threadgroup has. Four bits doubles the number of passes and buys a stable
scatter that fits.
"""

from max.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from max.gpu.memory import AddressSpace
from std.memory import stack_allocation

from ._bits import digit, ordered_bits_of_raw, unsigned_dtype

comptime BITS = 4
"""Digit width. See the module docstring for why it is not 8."""

comptime BUCKETS = 1 << BITS

comptime THREADS = 256
"""Threads per block."""

comptime ITEMS = 16
"""Items each thread owns, contiguously. Thread `t` owns the items at
`[t * ITEMS, (t + 1) * ITEMS)` of the tile, and that blocked arrangement is
what makes the rank below a stable one: earlier thread means earlier element."""

comptime TILE = THREADS * ITEMS


@always_inline
def _digit_of[
    D: DType
](raw: Scalar[unsigned_dtype[D]()], pass_index: Int) -> Int:
    """Returns the `pass_index`-th 4-bit digit of a raw key's ordered form."""
    return digit[D, BITS](ordered_bits_of_raw[D](raw), pass_index)


@always_inline
def _raw_view[
    D: DType, //
](keys: Pointer[Scalar[D], MutUntrackedOrigin]) -> Pointer[
    Scalar[unsigned_dtype[D]()], MutUntrackedOrigin
]:
    """Reinterprets a key buffer as unsigned integers of the same width.

    Every kernel reads and writes keys through this. Metal has no `double`:
    a kernel that so much as loads a `Float64` into a register fails to
    compile with "returns unsupported type 'double'". Moving the bits rather
    than the values sidesteps that, and costs nothing -- a sort never needs
    the numeric value, only the ordering its bits imply.
    """
    return keys.unsafe_bitcast[Scalar[unsigned_dtype[D]()]]()


def histogram_kernel[
    D: DType
](
    keys: Pointer[Scalar[D], MutUntrackedOrigin],
    block_counts: Pointer[UInt32, MutUntrackedOrigin],
    count: Int32,
    pass_index: Int32,
):
    """Counts each block's digits into `block_counts[block][bucket]`.

    Every thread counts its own items into its own column of
    `counts[bucket][thread]`, then one thread per bucket totals that bucket's
    row. No two threads ever write the same counter. The obvious alternative,
    sixteen shared counters bumped with `Atomic.fetch_add`, is correct but
    contended: on CUDA it made this kernel 60 times slower and ninety per cent
    of the whole sort.

    Parameters:
        D: The element type.

    Args:
        keys: The array being sorted, as it stands at the start of this pass.
        block_counts: Output, `grid_dim.x * BUCKETS` counters.
        count: How many elements the array holds.
        pass_index: Which digit this pass sorts on.
    """
    var counts = stack_allocation[
        BUCKETS * THREADS, UInt32, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    for b in range(BUCKETS):
        counts[unsafe_offset=b * THREADS + tid] = 0
    barrier()

    var n = Int(count)
    var which = Int(pass_index)
    var raw_keys = _raw_view(keys)
    var base = Int(block_idx.x) * TILE + tid * ITEMS
    for j in range(ITEMS):
        var index = base + j
        if index < n:
            var bucket = _digit_of[D](raw_keys[unsafe_offset=index], which)
            counts[unsafe_offset=bucket * THREADS + tid] += 1
    barrier()

    # One thread per bucket, totalling that bucket across threads.
    if tid < BUCKETS:
        var total = UInt32(0)
        for t in range(THREADS):
            total += counts[unsafe_offset=tid * THREADS + t]
        block_counts[unsafe_offset=Int(block_idx.x) * BUCKETS + tid] = total


def scan_kernel(
    block_counts: Pointer[UInt32, MutUntrackedOrigin],
    block_offsets: Pointer[UInt32, MutUntrackedOrigin],
    blocks: Int32,
):
    """Turns per-block counts into per-block starting positions.

    One thread per bucket. Each sums its bucket over every block to get the
    bucket's total, the totals are scanned to give each bucket its base in the
    output, and then each thread walks the blocks again handing out that
    bucket's share. Launched as a single block, because every bucket's base
    depends on every other bucket's total.

    Args:
        block_counts: `blocks * BUCKETS` counters from `histogram_kernel`.
        block_offsets: Output, where each block's share of each bucket starts.
        blocks: How many blocks the histogram pass used.
    """
    var totals = stack_allocation[
        BUCKETS, UInt32, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var block_count = Int(blocks)
    if tid >= BUCKETS:
        return

    var total = UInt32(0)
    for b in range(block_count):
        total += block_counts[unsafe_offset=b * BUCKETS + tid]
    totals[unsafe_offset=tid] = total
    barrier()

    # Sixteen counters, scanned by one thread. Parallelising this would cost
    # more in barriers than it saves.
    if tid == 0:
        var running = UInt32(0)
        for i in range(BUCKETS):
            var value = totals[unsafe_offset=i]
            totals[unsafe_offset=i] = running
            running += value
    barrier()

    var cursor = totals[unsafe_offset=tid]
    for b in range(block_count):
        block_offsets[unsafe_offset=b * BUCKETS + tid] = cursor
        cursor += block_counts[unsafe_offset=b * BUCKETS + tid]


def scatter_kernel[
    D: DType
](
    keys: Pointer[Scalar[D], MutUntrackedOrigin],
    out_keys: Pointer[Scalar[D], MutUntrackedOrigin],
    block_offsets: Pointer[UInt32, MutUntrackedOrigin],
    count: Int32,
    pass_index: Int32,
):
    """Moves each element to its place, stably.

    The rank an element needs is *how many elements of this block, before it,
    share its digit*. `ranks[bucket][thread]` is built as a per-thread count
    and then exclusive-scanned across threads, so it starts at exactly that
    number; walking a thread's own items in order does the rest. Blocked item
    assignment makes "before it" mean what it says.

    Parameters:
        D: The element type.

    Args:
        keys: The array as it stands at the start of this pass.
        out_keys: Output buffer, the same length.
        block_offsets: Where this block's share of each bucket starts.
        count: How many elements the array holds.
        pass_index: Which digit this pass sorts on.
    """
    var ranks = stack_allocation[
        BUCKETS * THREADS, UInt32, address_space=AddressSpace.SHARED
    ]()
    var bases = stack_allocation[
        BUCKETS, UInt32, address_space=AddressSpace.SHARED
    ]()

    var tid = Int(thread_idx.x)
    for b in range(BUCKETS):
        ranks[unsafe_offset=b * THREADS + tid] = 0
    barrier()

    var n = Int(count)
    var which = Int(pass_index)
    var raw_keys = _raw_view(keys)
    var raw_out = _raw_view(out_keys)
    var base = Int(block_idx.x) * TILE + tid * ITEMS

    for j in range(ITEMS):
        var index = base + j
        if index < n:
            var bucket = _digit_of[D](raw_keys[unsafe_offset=index], which)
            ranks[unsafe_offset=bucket * THREADS + tid] += 1
    barrier()

    # One thread per bucket, exclusive-scanning that bucket across threads.
    if tid < BUCKETS:
        var running = UInt32(0)
        for t in range(THREADS):
            var value = ranks[unsafe_offset=tid * THREADS + t]
            ranks[unsafe_offset=tid * THREADS + t] = running
            running += value
        bases[unsafe_offset=tid] = block_offsets[
            unsafe_offset=Int(block_idx.x) * BUCKETS + tid
        ]
    barrier()

    for j in range(ITEMS):
        var index = base + j
        if index < n:
            var value = raw_keys[unsafe_offset=index]
            var bucket = _digit_of[D](value, which)
            var slot = ranks[unsafe_offset=bucket * THREADS + tid]
            ranks[unsafe_offset=bucket * THREADS + tid] = slot + 1
            raw_out[
                unsafe_offset=Int(bases[unsafe_offset=bucket] + slot)
            ] = value
