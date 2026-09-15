"""The order-preserving bit mapping, shared with the CPU package.

The body of this file is a verbatim copy of
`mm_radix_sort/mm_radix_sort/_bits.mojo`, docstring aside. It is pure scalar
arithmetic with no host dependencies, so it compiles to device code unchanged
-- which is the single biggest reason a GPU port was cheap to attempt.

`ordered_bits_of_raw` is the entry point the kernels use. Metal has no
`double`: a kernel that so much as loads a `Float64` into a register fails to
compile. Every kernel therefore reads keys through a pointer bitcast to the
unsigned type of the same width and flips the raw bits, never materialising
the value.

It is copied rather than depended on only because `mm_radix_sort` is not
published yet. Once it is, this becomes a package dependency and the copy
goes away. Until then `test/test_bits.mojo` pins the exact bit patterns
this mapping must produce -- the same ones the CPU package's README
documents -- so the two copies cannot drift in behaviour unnoticed.
"""

from std.memory import bitcast
from std.sys.info import bit_width_of


@always_inline
def unsigned_dtype[D: DType]() -> DType:
    """Returns the unsigned integer `DType` of the same bit width as `D`.

    Parameters:
        D: The type whose width is to be matched.

    Returns:
        One of `uint8`, `uint16`, `uint32` or `uint64`.
    """
    comptime W = bit_width_of[D]()
    return (
        DType.uint8 if W
        == 8 else DType.uint16 if W
        == 16 else DType.uint32 if W
        == 32 else DType.uint64
    )


@always_inline
def ordered_bits_of_raw[
    D: DType
](raw: Scalar[unsigned_dtype[D]()]) -> Scalar[unsigned_dtype[D]()]:
    """Maps the raw bit pattern of a `D` onto a key that sorts in the same order.

    This is `ordered_bits` with the load already done. It exists because Metal
    has no `double`: a GPU kernel cannot so much as load a `Float64` into a
    register, so the GPU package reads keys through a pointer bitcast to the
    unsigned type and needs the flip to start from raw bits.

    Parameters:
        D: The type the bits came from.

    Args:
        raw: The bit pattern, reinterpreted as an unsigned integer.

    Returns:
        An unsigned integer of the same width whose unsigned ordering matches
        the natural ordering of `D`.
    """
    comptime U = unsigned_dtype[D]()
    comptime W = bit_width_of[D]()
    comptime SIGN = Scalar[U](1) << Scalar[U](W - 1)

    comptime if D.is_floating_point():
        # Arithmetic-shifting the sign bit down to all-ones (for a negative)
        # or all-zeros (for a positive), then forcing the sign bit on.
        var mask = (Scalar[U](0) - (raw >> Scalar[U](W - 1))) | SIGN
        return raw ^ mask
    elif D.is_signed():
        return raw ^ SIGN
    else:
        return raw


@always_inline
def ordered_bits[D: DType, //](value: Scalar[D]) -> Scalar[unsigned_dtype[D]()]:
    """Maps `value` onto an unsigned integer that sorts in the same order.

    Parameters:
        D: The element type.

    Args:
        value: The value to map.

    Returns:
        An unsigned integer of the same width whose unsigned ordering matches
        the natural ordering of `D`.
    """
    return ordered_bits_of_raw[D](bitcast[unsigned_dtype[D]()](value))


@always_inline
def digit[
    D: DType, BITS: Int
](key: Scalar[unsigned_dtype[D]()], pass_index: Int) -> Int:
    """Extracts the `pass_index`-th `BITS`-wide digit of an ordered key.

    Digits are numbered from the least significant end, so pass 0 sees the low
    `BITS` bits. The top digit of a key whose width is not a multiple of `BITS`
    is short, which is harmless: the bits above the width are always zero and
    the corresponding buckets stay empty.

    Parameters:
        D: The element type the key came from.
        BITS: The digit width.

    Args:
        key: An ordered key, as returned by `ordered_bits`.
        pass_index: Which digit to read.

    Returns:
        The digit, in `[0, 1 << BITS)`.
    """
    comptime U = unsigned_dtype[D]()
    comptime MASK = (Scalar[U](1) << Scalar[U](BITS)) - 1
    return Int((key >> Scalar[U](pass_index * BITS)) & MASK)


@always_inline
def pass_count[D: DType, BITS: Int]() -> Int:
    """Returns how many `BITS`-wide digits cover a value of type `D`.

    Parameters:
        D: The element type.
        BITS: The digit width.

    Returns:
        `ceil(bit_width_of[D]() / BITS)`.
    """
    return (bit_width_of[D]() + BITS - 1) // BITS
