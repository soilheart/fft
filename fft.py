import sys

import random

from numpy.fft import fft
from numpy import array

from pprint import pprint

import math
import cmath


def log2(val):
    return int(math.log(val, 2))


def numpy_variant(time):
    time_array = array(time)
    return list(fft(time_array))


def twiddle_factor(k, size):
    return cmath.exp(-2j*math.pi*k/size)


def bitreverse(val, bits):
    res = 0
    for bit in range(bits):
        if val & (1 << bit):
            res |= 1 << (bits - bit - 1);
    return res


def radix2_butterfly(op_a, op_b):
    res_a = op_a + op_b
    res_b = op_a - op_b
    return res_a, res_b


def dit_radix2(time):
    size = len(time)
    assert 1 << log2(size) == size
    bitreversed = [0] * size
    for i, val in enumerate(time):
        pos = bitreverse(i, log2(size))
        bitreversed[pos] = complex(val)

    result = bitreversed
    layers = log2(size)
    for layer in range(layers):
        butterfly_step = pow(2, layer)
        address_step = butterfly_step*2
        twiddle_step = size/address_step
        for butterfly in range(size/2):
            base_address = butterfly % butterfly_step + address_step * (butterfly // butterfly_step)
            # No twiddle factor
            op_a = result[base_address]
            tf_b = twiddle_factor((butterfly % butterfly_step) * twiddle_step, size)
            op_b = result[base_address + butterfly_step] * tf_b
            res_a, res_b = radix2_butterfly(op_a, op_b)
            result[base_address] = res_a
            result[base_address + butterfly_step] = res_b

    return result


def digitreverse(val, bits):
    res = 0
    assert bits % 2 == 0
    for digit in range(bits/2):
        digit_val = (val >> digit * 2) & 0x3
        res |= digit_val << (bits - 2 * digit - 2);
    return res


def radix4_butterfly(op_a, op_b, op_c, op_d):
    j = complex(0, 1)
    res_a = op_a + op_b + op_c + op_d
    res_b = op_a - j*op_b - op_c + j*op_d
    res_c = op_a - op_b + op_c - op_d
    res_d = op_a + j*op_b - op_c - j*op_d
    return res_a, res_b, res_c, res_d


def dit_radix4(time):
    size = len(time)
    assert 1 << log2(size) == size
    digitreversed = [0] * size
    for i, val in enumerate(time):
        pos = digitreverse(i, log2(size))
        digitreversed[pos] = complex(val)

    result = digitreversed
    layers = log2(size)/2
    for layer in range(layers):
        butterfly_step = pow(4, layer)
        address_step = butterfly_step*4
        twiddle_step = size/address_step
        for butterfly in range(size/4):
            base_address = butterfly % butterfly_step + address_step * (butterfly // butterfly_step)
            tfs = [twiddle_factor((butterfly % butterfly_step) * twiddle_step * tf, size) for tf in range(4)]
            ops = [result[base_address + i*butterfly_step] * tfs[i] for i in range(4)]
            res_a, res_b, res_c, res_d = radix4_butterfly(*ops)
            result[base_address + 0*butterfly_step] = res_a
            result[base_address + 1*butterfly_step] = res_b
            result[base_address + 2*butterfly_step] = res_c
            result[base_address + 3*butterfly_step] = res_d

    return result


def reverse(val, bits, base=4):
    res = 0
    assert 1 << log2(base) == base
    bits_per_digit = log2(base)
    assert bits % log2(base) == 0
    for digit in range(bits/bits_per_digit):
        digit_val = (val >> digit * bits_per_digit) & ((1 << bits_per_digit) - 1)
        res |= digit_val << (bits - bits_per_digit * digit - bits_per_digit);
    return res


def butterfly_op(ops, radix):
    assert len(ops) == radix
    if radix == 2:
        return radix2_butterfly(*ops)
    elif radix == 4:
        return radix4_butterfly(*ops)
    assert False


def dit(time, radix):
    size = len(time)
    assert 1 << log2(size) == size
    time_reversed = [0] * size
    for i, val in enumerate(time):
        pos = reverse(i, log2(size), radix)
        time_reversed[pos] = complex(val)

    result = time_reversed
    layers = 2*log2(size)/radix
    for layer in range(layers):
        butterfly_step = pow(radix, layer)
        address_step = butterfly_step*radix
        twiddle_step = size/address_step
        for butterfly in range(size/radix):
            base_address = butterfly % butterfly_step + address_step * (butterfly // butterfly_step)
            tfs = [twiddle_factor((butterfly % butterfly_step) * twiddle_step * tf, size) for tf in range(radix)]
            ops = [result[base_address + i*butterfly_step] * tfs[i] for i in range(radix)]
            for i, res in enumerate(butterfly_op(ops, radix)):
                result[base_address + i*butterfly_step] = res

    return result


def main(argv):
    random.seed(123)
    length = 8
    time = [random.random() for _ in range(16)]

    print("Time:")
    pprint(time)

    print("Numpy:")
    pprint(numpy_variant(time))

    print("DIT radix2:")
    pprint(dit_radix2(time))
    pprint(dit(time, 2))

    print("DIT radix4:")
    pprint(dit_radix4(time))
    pprint(dit(time, 4))


if __name__ == "__main__":
    main(sys.argv)
