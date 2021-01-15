from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit, ClassicalRegister
from math import pi
from qiskit import Aer, execute
from utils import modinv, post_process


# `n` has to be at least 2
# e[1] is toggled when there is an overflow and x[0]
# e[2] is toggled when there is no overflow and x[0]
def carry_dirty(n: int, v: int):
    a = QuantumRegister(n, "a")
    x = QuantumRegister(n, "x")
    e = QuantumRegister(3, "e")
    circ = QuantumCircuit(a, x, e)

    # flips bit g[i] to indicate carry
    def forward(i: int):
        if i > 1:
            if v & 1 << i:
                circ.cx(a[i], x[i])
                circ.x(a[i])
            circ.ccx(x[i - 1], a[i], x[i])
            forward(i - 1)
            circ.ccx(x[i - 1], a[i], x[i])
        else:
            if v & 2:
                circ.cx(a[1], x[1])
                circ.x(a[1])
            if v & 1:
                circ.ccx(a[0], a[1], x[1])

    # undoes the previous method
    def backward(i: int):
        if i > 1:
            circ.ccx(x[i - 1], a[i], x[i])
            backward(i - 1)
            circ.ccx(x[i - 1], a[i], x[i])
            if v & 1 << i:
                circ.x(a[i])
                circ.cx(a[i], x[i])
        else:
            if v & 1:
                circ.ccx(a[0], a[1], x[1])
            if v & 2:
                circ.x(a[1])
                circ.cx(a[1], x[1])

    # the next gates combines with the ones bellow to condition on whether x[n - 1] is flipped
    circ.ccx(x[0], x[n - 1], e[1])
    circ.x(x[n - 1])
    circ.ccx(x[0], x[n - 1], e[2])
    circ.x(x[n - 1])

    forward(n - 1)

    circ.ccx(x[0], x[n - 1], e[1])
    circ.ccx(x[0], x[n - 1], e[2])

    backward(n - 1)

    return circ


# applies the fourier transform
def qft(n: int):
    a = QuantumRegister(n, "a")
    circ = QuantumCircuit(a)

    for i in reversed(range(n)):
        circ.h(a[i])
        for j in range(i):
            circ.cp(pi / 2 ** (i - j), a[j], a[i])

    return circ


# adds one or two constants based on e[1] and e[2]
def fourier_add_const(n: int, v1: int, v2: int):
    a = QuantumRegister(n, "a")
    e = QuantumRegister(3, "e")
    circ = QuantumCircuit(a, e)

    circ.extend(qft(n))
    for i in range(n):
        circ.cp(pi * v1 / (1 << i), e[1], a[i])
        circ.cp(pi * v2 / (1 << i), e[2], a[i])
    circ.extend(qft(n).inverse())

    return circ


# adds a constants `v` modulo `N` conditioned on x[0]
def fourier_modulo_add(n: int, v: int, N: int):
    a = QuantumRegister(n, "a")
    x = QuantumRegister(n, "x")
    e = QuantumRegister(3, "e")
    circ = QuantumCircuit(a, x, e)

    circ.extend(carry_dirty(n, v - N))  # set e[1] and e[2]
    circ.extend(fourier_add_const(n, v - N, v))
    circ.extend(carry_dirty(n, -v))  # clean up e[1] and e[2]
    circ.cx(x[0], e[1])
    circ.cx(x[0], e[2])

    return circ


# calculates x = v * x % N if e[0]
# assumes a and e[1] and e[2] are zeroed
def modulo_mul(n: int, v: int, N: int) -> QuantumCircuit:
    a = QuantumRegister(n, "a")
    x = QuantumRegister(n, "x")
    e = QuantumRegister(3, "e")
    circ = QuantumCircuit(a, x, e)

    v_inv = modinv(v, N)

    # swap `a` and `x` if not e[0]
    circ.x(e[0])
    for i in range(n):
        circ.cswap(e[0], a[i], x[i])
    circ.x(e[0])
    # calculate `a = x * v % N`
    for i in range(n):
        if i != 0:
            circ.swap(x[i], x[0])
        circ.extend(fourier_modulo_add(n, (v << i) % N, N))
        if i != 0:
            circ.swap(x[i], x[0])
    # swap `a` and `x` if e[0]
    for i in range(n):
        circ.cswap(e[0], a[i], x[i])
    # calculate `a = 0`
    for i in range(n):
        if i != 0:
            circ.swap(x[i], x[0])
        circ.extend(fourier_modulo_add(n, (v_inv << i) % N, N).inverse())
        if i != 0:
            circ.swap(x[i], x[0])
    # swap `a` and `x` if not e[0]
    circ.x(e[0])
    for i in range(n):
        circ.cswap(e[0], a[i], x[i])
    circ.x(e[0])

    return circ


if __name__ == "__main__":
    n = 4
    a = QuantumRegister(n, "a")
    x = QuantumRegister(n, "x")
    e = QuantumRegister(3, "e")
    r = ClassicalRegister(n, "r")
    circ = QuantumCircuit(a, x, e, r)
    circ.x(x[0])  # set `x` to 1
    circ.x(e[0])  # enable multiplication
    circ.extend(modulo_mul(n, 2, 15))
    circ.extend(modulo_mul(n, 2, 15))
    circ.extend(modulo_mul(n, 2, 15))
    circ.extend(modulo_mul(n, 2, 15))
    # circ.extend(fourier_modulo_add(n, 7, 15))
    # circ.extend(carry_dirty(n, -2))
    circ.measure(x, r)

    simulator = Aer.get_backend("qasm_simulator")
    result = execute(circ, simulator).result().get_counts(circ)
    print(result)

    # print(circ.draw("text"))

