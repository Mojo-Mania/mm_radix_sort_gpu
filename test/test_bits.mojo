"""Pins the order-preserving mapping, so the copy cannot drift from the CPU one.

`mm_radix_sort_gpu/_bits.mojo` is a copy of the CPU package's file, and a copy
can be edited. What stops the two diverging silently is this: the exact bit
patterns the mapping is required to produce, which are the same ones the CPU
package's README documents. If either file changes what it computes, one of
these fails.

Needs no GPU -- it is the one test in this package that does not.
"""

from mm_radix_sort_gpu._bits import (
    digit,
    ordered_bits,
    ordered_bits_of_raw,
    pass_count,
    unsigned_dtype,
)
from std.memory import bitcast
from std.testing import TestSuite, assert_equal, assert_true

# Every helper below returns a *concrete* width. `Scalar[unsigned_dtype[D]()]`
# is an unfolded parametric expression: it will not unify with a plain
# `UInt32` at a call site, and two of them built from different source types
# will not unify with each other either.


@always_inline
def _ordered8[D: DType, //](value: Scalar[D]) -> UInt8:
    """The mapping, taking a value, as the CPU sort calls it."""
    return rebind[UInt8](ordered_bits(value))


@always_inline
def _ordered32[D: DType, //](value: Scalar[D]) -> UInt32:
    """The mapping, taking a value, as the CPU sort calls it."""
    return rebind[UInt32](ordered_bits(value))


@always_inline
def _ordered64[D: DType, //](value: Scalar[D]) -> UInt64:
    """The mapping, taking a value, as the CPU sort calls it."""
    return rebind[UInt64](ordered_bits(value))


@always_inline
def _raw16[D: DType, //](value: Scalar[D]) -> UInt16:
    """The mapping, taking raw bits, as the kernels call it."""
    comptime U = unsigned_dtype[D]()
    return rebind[UInt16](ordered_bits_of_raw[D](bitcast[U](value)))


@always_inline
def _raw32[D: DType, //](value: Scalar[D]) -> UInt32:
    """The mapping, taking raw bits, as the kernels call it."""
    comptime U = unsigned_dtype[D]()
    return rebind[UInt32](ordered_bits_of_raw[D](bitcast[U](value)))


@always_inline
def _raw64[D: DType, //](value: Scalar[D]) -> UInt64:
    """The mapping, taking raw bits, as the kernels call it."""
    comptime U = unsigned_dtype[D]()
    return rebind[UInt64](ordered_bits_of_raw[D](bitcast[U](value)))


def test_unsigned_dtype_widths() raises:
    comptime FROM_INT8 = unsigned_dtype[DType.int8]()
    comptime FROM_INT16 = unsigned_dtype[DType.int16]()
    comptime FROM_FLOAT32 = unsigned_dtype[DType.float32]()
    comptime FROM_FLOAT64 = unsigned_dtype[DType.float64]()
    assert_true(FROM_INT8 == DType.uint8, "int8 -> uint8")
    assert_true(FROM_INT16 == DType.uint16, "int16 -> uint16")
    assert_true(FROM_FLOAT32 == DType.uint32, "float32 -> uint32")
    assert_true(FROM_FLOAT64 == DType.uint64, "float64 -> uint64")


def test_float32_bit_patterns() raises:
    """The published Float32 table. Raw does not ascend; ordered does."""
    assert_equal(_ordered32(Float32(-2.0)), UInt32(0x3FFFFFFF), "-2.0")
    assert_equal(_ordered32(Float32(-1.0)), UInt32(0x407FFFFF), "-1.0")
    assert_equal(_ordered32(Float32(-0.0)), UInt32(0x7FFFFFFF), "-0.0")
    assert_equal(_ordered32(Float32(0.0)), UInt32(0x80000000), "0.0")
    assert_equal(_ordered32(Float32(1.0)), UInt32(0xBF800000), "1.0")
    assert_equal(_ordered32(Float32(2.0)), UInt32(0xC0000000), "2.0")


def test_int8_bit_patterns() raises:
    """The published Int8 table."""
    assert_equal(_ordered8(Int8(-128)), UInt8(0), "-128")
    assert_equal(_ordered8(Int8(-1)), UInt8(127), "-1")
    assert_equal(_ordered8(Int8(0)), UInt8(128), "0")
    assert_equal(_ordered8(Int8(127)), UInt8(255), "127")


def test_unsigned_is_the_identity() raises:
    assert_equal(_ordered8(UInt8(7)), UInt8(7), "uint8")
    assert_equal(_ordered32(UInt32(123456)), UInt32(123456), "uint32")
    assert_equal(_ordered64(UInt64.MAX), UInt64.MAX, "uint64 max")


def test_raw_form_agrees_with_the_value_form() raises:
    """The GPU reads bits, the CPU reads values; they must agree exactly."""
    for i in range(-2000, 2000):
        var single = Float32(Float64(i) * 0.25)
        assert_equal(
            _ordered32(single), _raw32(single), String("float32 at ", i)
        )
        var double = Float64(i) * 0.25
        assert_equal(
            _ordered64(double), _raw64(double), String("float64 at ", i)
        )
        var signed = Int32(i)
        assert_equal(_ordered32(signed), _raw32(signed), String("int32 at ", i))


def test_mapping_preserves_order() raises:
    """The property the whole package rests on, checked rather than assumed."""
    var floats = [
        Float64(-1e308),
        -1.0,
        -5e-324,
        -0.0,
        5e-324,
        1.0,
        1e308,
    ]
    for i in range(1, len(floats)):
        assert_true(
            _raw64(floats[i - 1]) <= _raw64(floats[i]),
            String("float64 order broken at ", i),
        )
    for i in range(-3000, 3000):
        assert_true(
            _raw16(Int16(i - 1)) < _raw16(Int16(i)),
            String("int16 order broken at ", i),
        )


def test_digit_and_pass_count() raises:
    comptime PASSES32 = pass_count[DType.uint32, 4]()
    comptime PASSES64 = pass_count[DType.uint64, 4]()
    comptime PASSES8 = pass_count[DType.uint8, 4]()
    assert_equal(PASSES32, 8, "uint32 at 4 bits")
    assert_equal(PASSES64, 16, "uint64 at 4 bits")
    assert_equal(PASSES8, 2, "uint8 at 4 bits")

    var key = ordered_bits(UInt32(0xABCD1234))
    assert_equal(digit[DType.uint32, 4](key, 0), 4, "low digit")
    assert_equal(digit[DType.uint32, 4](key, 1), 3, "next digit")
    assert_equal(digit[DType.uint32, 4](key, 7), 0xA, "top digit")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
