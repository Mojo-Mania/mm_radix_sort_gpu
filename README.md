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
sort is. The crossover measured here is n ≈ 256 Ki.

**The headline number is not the interesting one.** Against the stdlib's
`sort` this is 45x faster on 16 Mi `uint32`. Against a *tuned single-threaded
radix sort on the same machine* it is **2.2x**. Most of that 45x is radix
beating comparison, which you can have on the CPU without a GPU at all.

**Unified memory is why the crossover is so low.** There is no host-to-device
transfer to amortise: on Apple silicon the buffer the GPU sorts is the same
memory the CPU wrote. On a discrete GPU that transfer usually decides whether
shipping a sort off the CPU is worth it at all, and it would move the
crossover up by an order of magnitude.

**NaN is not handled.** A NaN has no place in a total order.

## Install

```toml
[dependencies]
mm_radix_sort_gpu = { git = "https://github.com/Mojo-Mania/mm_radix_sort_gpu.git" }
```

Needs the `max` package for `max.gpu`, and a device to run on. Developed
against an Apple M4 Max through Metal; the kernels use nothing vendor-specific
beyond threadgroup memory, barriers and atomics, but they have not been run on
NVIDIA or AMD hardware.

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

1. **histogram** — each block counts its tile's digits into threadgroup memory
   and writes `block_counts[block][bucket]`.
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

## Development

```bash
pixi run test       # both suites (16 tests) -- 9 of them need a GPU
pixi run test-bits  # just the 7 that do not
pixi run main       # the example -- needs a GPU
pixi run bench      # the table above -- needs a GPU
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
