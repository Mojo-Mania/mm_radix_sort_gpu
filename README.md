# mm_radix_sort_gpu

[![CI](https://github.com/Mojo-Mania/mm_radix_sort_gpu/actions/workflows/ci.yml/badge.svg)](https://github.com/Mojo-Mania/mm_radix_sort_gpu/actions/workflows/ci.yml)

A least-significant-digit radix sort for [Mojo](https://mojolang.org) that
runs on the GPU. The device half of
[mm_radix_sort](https://github.com/Mojo-Mania/mm_radix_sort).

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
same order-preserving bit mapping the CPU package uses. The sort is stable and
runs entirely on the device.

## When to use it

**Above about a quarter of a million elements.** Below that the CPU radix sort
in `mm_radix_sort` is faster, and below a thousand or so a plain comparison
sort is. The crossover measured on an Apple M4 Max is n ≈ 256 Ki. On the
discrete laptop GPU measured further down it is between 256 Ki and 1 Mi, and
about 1 Mi for `uint64`.

**The headline number is not the interesting one.** Against the stdlib's
`sort` this is 45x faster on 16 Mi `uint32`. Against a *tuned single-threaded
radix sort on the same machine* it is **2.2x**. Most of that 45x is radix
beating comparison, which you can have on the CPU without a GPU at all.

**Unified memory is why the crossover is so low.** There is no host-to-device
transfer to amortise: on Apple silicon the buffer the GPU sorts is the same
memory the CPU wrote. On a discrete GPU that transfer usually decides whether
shipping a sort off the CPU is worth it at all. The RTX 4050 numbers below do
not count that transfer, so they are a best case for a discrete GPU.

**NaN is not handled.** A NaN has no place in a total order.

## Install

```toml
[dependencies]
mm_radix_sort_gpu = { git = "https://github.com/Mojo-Mania/mm_radix_sort_gpu.git" }
```

Needs the `max` package for `max.gpu`, and a device to run on. Developed
against an Apple M4 Max through Metal; the kernels use nothing vendor-specific
beyond threadgroup memory and barriers. They also run on NVIDIA
through CUDA: all 16 tests pass on an RTX 4050 Laptop GPU. They have not been
run on AMD hardware.

## API

```mojo
gpu_radix_sort(ctx, keys, count)
```

Sorts the first `count` elements of the device buffer `keys` in ascending
order, in place. Work is *enqueued*, not awaited — call `ctx.synchronize()`
before reading the result.

It allocates two scratch buffers: one the length of the keys, and one holding
`ceil(count / 4096) * 16` counters twice over. It makes
`ceil(bit_width_of[D]() / 4)` passes — 8 for a 32-bit type, 16 for a 64-bit
one.

## How it works

A pass is three kernels: **histogram**, **scan**, **scatter**. That is the same
shape as the CPU package's `lsb_radix_sort`, with the prefix sum split in two
because blocks cannot see each other.

1. **histogram** — each thread counts its own items into its own column of a
   `BUCKETS × threads` table in threadgroup memory, then one thread per bucket
   totals its row into `block_counts[block][bucket]`. No counter is shared, so
   there is nothing to synchronise. Sixteen shared counters bumped atomically
   would be simpler, but on CUDA that contention was 90% of the sort.
2. **scan** — one block, one thread per bucket. Each thread totals its bucket
   across every block, the totals are scanned to give each bucket its base in
   the output, then each thread walks the blocks again handing out that
   bucket's share. This is what tells block 7 where its share of bucket 3
   begins.
3. **scatter** — each block ranks its own elements and writes them out.

### Why the digit is 4 bits and not 8

An LSD sort is only correct if every pass is **stable**: pass *k* must not
disturb the order that passes 0..*k*-1 established among elements sharing digit
*k*. On a CPU that is free — the scatter walks the array in order. On a GPU
every element has to be told its rank among the elements of its block that
share its digit, and that rank comes from a per-thread histogram in
threadgroup memory.

That histogram costs `BUCKETS × threads` counters. At 4 bits and 256 threads
that is 8 KiB, which fits. At 8 bits it is 128 KiB, which no threadgroup has.
Four bits doubles the pass count and buys a stable scatter that fits — and the
measurements below say the trade is worth taking.

Items are assigned to threads in **blocked** order — thread *t* owns items
`[t·16, (t+1)·16)` — so "an earlier thread" means "an earlier element", which
is what makes the exclusive scan across threads produce a stable rank.

### Metal has no `double`

A kernel that so much as loads a `Float64` into a register fails to compile:
`instruction ... returns unsupported type 'double'`. The sort therefore reads
and writes keys through a pointer bitcast to the unsigned integer of the same
width, and flips the raw bits rather than the value. That costs nothing — a
sort never needs a number, only the ordering its bits imply — and `Float64`
works as a result.

This is the one place the GPU port needed a change to the shared bit mapping:
`ordered_bits_of_raw` takes the bits with the load already done.

## Performance

### Apple M4 Max (unified memory, Metal)

> These numbers predate the change to `histogram_kernel` that removed its
> atomic counters (see the RTX 4050 section below). They have not yet been
> re-measured with it.

Apple M4 Max, `-D ASSERT=none`, min of 20 runs. Every timing restores the
working buffer from a pristine device buffer before sorting, inside the timed
region, so no run measures already-sorted input. That restore is reported as a
floor rather than subtracted out. Output is checked against the stdlib's
`sort`, not merely checked for being ascending.

Nanoseconds per element. Reproduce with `pixi run bench`.

| type | n | copy floor | **gpu** | cpu `lsb[11]` | host `sort` | vs cpu radix | vs `sort` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uint32` | 64 Ki | 1.53 | 6.15 | **2.19** | 33.9 | 0.36x | 5.5x |
| `uint32` | 256 Ki | 0.33 | **2.01** | 2.12 | 39.6 | 1.05x | 19.7x |
| `uint32` | 1 Mi | 0.07 | **1.23** | 2.19 | 45.3 | 1.78x | 36.9x |
| `uint32` | 4 Mi | 0.03 | **1.16** | 2.51 | 48.6 | 2.16x | 41.8x |
| `uint32` | 16 Mi | 0.02 | **1.19** | 2.64 | 53.4 | 2.22x | 44.9x |
| `float32` | 1 Mi | 0.13 | **1.52** | 2.54 | 55.4 | 1.67x | 36.5x |
| `float32` | 16 Mi | 0.02 | **1.19** | 2.85 | 67.0 | 2.40x | 56.5x |
| `uint64` | 1 Mi | 0.10 | **2.73** | 4.78 | 45.5 | 1.75x | 16.6x |
| `uint64` | 16 Mi | 0.04 | **2.80** | 6.32 | 53.8 | 2.26x | 19.2x |

The `cpu lsb[11]` column is `mm_radix_sort`'s `lsb_radix_sort[BITS=11]` on one
core of the same machine, measured at the same sizes.

Three things worth reading off it.

**The crossover against the CPU radix sort is around 256 Ki.** Below that the
fixed cost of launching three kernels per pass — eight passes for a 32-bit
type — dominates. At 64 Ki the GPU is nearly three times *slower*.

**At scale the GPU wins by about 2.2x, not by 45x.** Both numbers are in the
table and the second one is the one people quote, but a tuned radix sort on
one CPU core is already within a factor of three of a whole GPU. Radix sort is
memory-bound, and on this machine both processors are reading the same
memory at similar bandwidth.

**`uint64` costs 16 passes and it shows** — 2.8 ns/element against 1.19 for a
32-bit type, a little over double for double the passes.

### NVIDIA RTX 4050 Laptop GPU (discrete, CUDA)

AMD Ryzen AI 9 HX 370 host with an NVIDIA GeForce RTX 4050 Laptop GPU (6 GiB,
PCIe 4.0 x8), Arch Linux 7.1.9, driver 610.57.04, Mojo 1.2.0.dev2026091505, MAX
26.7.0.dev2026091505. On AC power, `performance` platform profile; the GPU sat
in P0 at 100% utilisation with no throttle reasons active. Same method as above:
`-D ASSERT=none`, min of 20 runs, output checked against `sort`. Each figure is
the better of two full `pixi run bench` runs, which agreed to within 1%.

The buffers are created and filled once, outside the timed region, so **no
host-to-device or device-to-host transfer is included**. On this machine that
transfer is real PCIe traffic, and it would only add to the GPU column.

| type | n | copy floor | gpu | cpu `lsb[11]` | host `sort` | vs cpu radix | vs `sort` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uint32` | 64 Ki | 0.09 | 3.10 | **1.89** | 37.9 | 0.61x | 12.2x |
| `uint32` | 256 Ki | 0.03 | 2.12 | **1.95** | 43.4 | 0.92x | 20.5x |
| `uint32` | 1 Mi | 0.01 | **1.67** | 1.97 | 49.4 | 1.18x | 29.6x |
| `uint32` | 4 Mi | 0.03 | **1.45** | 4.74 | 52.9 | 3.27x | 36.5x |
| `uint32` | 16 Mi | 0.05 | **1.32** | 5.30 | 58.2 | 4.00x | 44.0x |
| `float32` | 64 Ki | 0.09 | 3.13 | **2.30** | 45.3 | 0.73x | 14.5x |
| `float32` | 256 Ki | 0.03 | **2.13** | 2.43 | 51.5 | 1.14x | 24.2x |
| `float32` | 1 Mi | 0.01 | **1.68** | 2.45 | 56.6 | 1.46x | 33.8x |
| `float32` | 4 Mi | 0.03 | **1.45** | 5.05 | 62.7 | 3.49x | 43.3x |
| `float32` | 16 Mi | 0.05 | **1.33** | 5.70 | 68.8 | 4.29x | 51.9x |
| `uint64` | 64 Ki | 0.11 | 7.81 | **2.51** | 38.0 | 0.32x | 4.9x |
| `uint64` | 256 Ki | 0.04 | 5.39 | **2.56** | 43.5 | 0.47x | 8.1x |
| `uint64` | 1 Mi | 0.02 | 4.25 | **4.21** | 50.6 | 0.99x | 11.9x |
| `uint64` | 4 Mi | 0.08 | **3.91** | 7.42 | 54.0 | 1.90x | 13.8x |
| `uint64` | 16 Mi | 0.10 | **4.03** | 7.69 | 58.8 | 1.91x | 14.6x |

The `cpu lsb[11]` column is `mm_radix_sort`'s `lsb_radix_sort[BITS=11]` on one
core of the Ryzen. It was measured on the bench's exact inputs (`seed(1)`, same
value distribution), at the same sizes, with the same min-of-20 and a `memcpy`
restore inside the timed region. `mm_radix_sort`'s own `pixi run bench` uses
different sizes and reports a mean, so it will not reproduce this column as-is.

What this table says:

**The crossover against the CPU radix sort is between 256 Ki and 1 Mi.** It
comes at 256 Ki for `float32`, just after it for `uint32`, and at about 1 Mi
for `uint64`. Below that the fixed cost of the kernel launches dominates, as
on the M4 Max.

**At scale the GPU wins by about 4x for 32-bit keys and 1.9x for `uint64`.**
That margin is wider than the M4 Max's 2.2x, and the GPU is not the reason: at
16 Mi it is only about 10% slower than the M4 Max GPU. What differs is the CPU. The Zen 5
core's radix sort slows from about 2 ns to 5 ns per element between 1 Mi and
4 Mi as the working set outgrows cache, and the M4 Max core's barely does.

**`uint64` costs three times `uint32` here**, 4.03 against 1.32 ns per element
for double the passes. On the M4 Max the ratio is 2.4x. Why it is steeper here
has not been investigated.

**Before the histogram fix this GPU was ten times slower.** The first version
of `histogram_kernel` counted into sixteen counters shared by all 256 threads
of a block, bumped with `Atomic.fetch_add`. On CUDA that took 13 ns per element
and 90% of the sort, and the GPU lost to the CPU radix sort at every size, by
up to 16x. Timing each kernel with a `synchronize()` between them found it:
scatter and scan together cost about 1 ns. Barriers and threadgroup memory
cost almost nothing (0.07 ns per element per launch), but the atomic
increments did not.

## Development

```bash
pixi run test       # both suites (16 tests) -- 9 of them need a GPU
pixi run test-bits  # just the 7 that do not
pixi run main       # the example -- needs a GPU
pixi run bench      # the tables above -- needs a GPU
pixi run format     # mojo format
pixi run docs       # docstring check
pixi build          # the conda package (needs pixi >= 0.80)
```

GitHub's hosted runners have no GPU, so CI runs `test-bits`, `format`, `docs`
and `build`. That does check the one thing a copied file most needs checking:
`test/test_bits.mojo` pins the exact bit patterns the mapping must produce, so
this package's copy of `_bits.mojo` cannot drift from the CPU package's
original without failing. The nine device tests are run locally before
pushing, and take about a second.

## What is not here

**MSD sorts.** `mm_radix_sort`'s `msb_radix_sort` and `american_flag_sort`
recurse into unevenly sized buckets, which is load imbalance, and the American
flag permutation is a pointer-chasing cycle, which is inherently serial.
Neither belongs on a GPU.

**String sorting.** `byte_radix_sort` is variable-length keys and irregular
recursion. The design that would work is a hybrid: pack the first eight bytes
of each key into a `UInt64`, sort those here, resolve ties on the host. The
book corpus in `mm_radix_sort` says tokens diverge after about 4.5 bytes, so
eight would settle most of them in one pass.

**A parallel inter-block scan.** `scan_kernel` is a single block with one
thread per bucket walking every block's counts serially. It has not shown up
as a bottleneck at the sizes measured, but it is O(blocks) on 16 threads and
will.

## Provenance

The device half of [mm_radix_sort](https://github.com/Mojo-Mania/mm_radix_sort),
which is a port of the `radix_sorting/` directory of
[mzaks/mojo-sort](https://github.com/mzaks/mojo-sort).

`mm_radix_sort_gpu/_bits.mojo` is a copy of the CPU package's file rather than
a dependency on it, only because that package is not published yet.

## License

MIT. See [LICENSE](LICENSE).
