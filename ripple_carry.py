from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit, ClassicalRegister
from qiskit import Aer, execute
from math import pi, log, ceil
from collections import defaultdict
from utils import modinv, post_process, factor_finder


def carry(n: int, v: int) -> QuantumCircuit:
    """
    assert c == 0

    if e[1]:
        e[2] ^= (a + v)[n]  # flip e[2] if there is an overflow 
    """
    a = QuantumRegister(n, "a")
    c = QuantumRegister(n, "c")
    e = QuantumRegister(3, "e")
    circ = QuantumCircuit(a, c, e)

    if v & 1:
        circ.cx(a[0], c[0])

    for i in range(1, n):
        if v & 1 << i:
            circ.cx(a[i], c[i])
            circ.x(a[i])
        circ.ccx(c[i - 1], a[i], c[i])

    circ.ccx(c[-1], e[1], e[2])

    for i in reversed(range(1, n)):
        circ.ccx(c[i - 1], a[i], c[i])
        if v & 1 << i:
            circ.x(a[i])
            circ.cx(a[i], c[i])

    if v & 1:
        circ.cx(a[0], c[0])

    return circ


def add_const(n: int, v1: int, v2: int) -> QuantumCircuit:
    """
    assert c[:-2] == 0
    assert not (c[-2] and c[-1])

    if c[-2]:
        a += v1
    if c[-1]:
        a += v2
    """
    a = QuantumRegister(n, "a")
    c = QuantumRegister(n, "c")
    circ = QuantumCircuit(a, c)

    c1, c2 = c[-2:]  # two extra control bits
    c = c[:-2] + a[-1:]
    last = len(c) - 1

    if v1 & 1:
        circ.ccx(c1, a[0], c[0])
    if v2 & 1:
        circ.ccx(c2, a[0], c[0])

    for i in range(1, len(c)):
        if v1 & 1 << i:
            circ.ccx(c1, a[i], c[i])
            circ.cx(c1, a[i])
        if v2 & 1 << i:
            circ.ccx(c2, a[i], c[i])
            circ.cx(c2, a[i])
        circ.ccx(c[i - 1], a[i], c[i])

    if v1 & 1 << (last + 1):
        circ.cx(c1, a[last + 1])  # this is c[last]
    if v2 & 1 << (last + 1):
        circ.cx(c2, a[last + 1])  # this is c[last]
    circ.cx(c[last - 1], a[last])

    for i in reversed(range(1, last)):
        circ.ccx(c[i - 1], a[i], c[i])
        if v1 & 1 << i:
            circ.cx(c1, a[i])
            circ.ccx(c1, a[i], c[i])
            circ.cx(c1, a[i])
        if v2 & 1 << i:
            circ.cx(c2, a[i])
            circ.ccx(c2, a[i], c[i])
            circ.cx(c2, a[i])
        circ.cx(c[i - 1], a[i])

    if v1 & 1:
        circ.ccx(c1, a[0], c[0])
        circ.cx(c1, a[0])
    if v2 & 1:
        circ.ccx(c2, a[0], c[0])
        circ.cx(c2, a[0])

    return circ


def modulo_add(n: int, v: int, N: int) -> QuantumCircuit:
    """
    assert c == 0
    assert not e[2]

    if e[1]:
        a = (a + v) % N
    """
    a = QuantumRegister(n, "a")
    c = QuantumRegister(n, "c")
    e = QuantumRegister(3, "e")
    circ = QuantumCircuit(a, c, e)

    circ.extend(carry(n, v - N))  # set e[2]
    circ.ccx(e[1], e[2], c[-2])
    circ.x(e[2])
    circ.ccx(e[1], e[2], c[-1])
    circ.extend(add_const(n, v - N, v))
    circ.ccx(e[1], e[2], c[-1])
    circ.x(e[2])
    circ.ccx(e[1], e[2], c[-2])
    circ.extend(carry(n, -v))  # clean up e[2]
    circ.cx(e[1], e[2])

    return circ


def modulo_mul(n: int, v: int, N: int) -> QuantumCircuit:
    """
    assert a == 0
    assert c == 0
    assert not e[1] and not e[2]

    if e[0]:
        x = (v * x) % N
    """
    a = QuantumRegister(n, "a")
    c = QuantumRegister(n, "c")
    x = QuantumRegister(n, "x")
    e = QuantumRegister(3, "e")
    circ = QuantumCircuit(a, c, x, e)

    v_inv = modinv(v, N)

    for i in range(n):
        circ.ccx(e[0], x[i], e[1])
        circ.extend(modulo_add(n, (v << i) % N, N))
        circ.ccx(e[0], x[i], e[1])
    for i in range(n):
        circ.cswap(e[0], a[i], x[i])
    for i in range(n):
        circ.ccx(e[0], x[i], e[1])
        circ.extend(modulo_add(n, (v_inv << i) % N, N).inverse())
        circ.ccx(e[0], x[i], e[1])

    return circ


def modulo_exp(n: int, v: int, exp: int, N: int):
    """
    assert a == 0
    assert c == 0
    assert x == 0
    assert not e[0] and not e[1] and not e[2]

    x = (v ** exp) % N
    r = x 
    """
    x = QuantumRegister(n, "x")
    a = QuantumRegister(n, "a")
    c = QuantumRegister(n, "c")
    e = QuantumRegister(3, "e")
    r = ClassicalRegister(n, "r")
    circ = QuantumCircuit(x, a, c, e, r)
    circ.x(x[0])  # start with value 1

    for i in range(2 * n):
        if exp & 1 << i:
            circ.x(e[0])
        circ.extend(modulo_mul(n, v ** (1 << i), N))
        if exp & 1 << i:
            circ.x(e[0])

    circ.measure(x, r)
    return circ


def quantum_period(n: int, v: int, N: int) -> QuantumCircuit:
    """
    returns a number close to `N * k / period` where `k` is an integer.
    `e[0]` is the control qubit which is used for the conditional modular multiplication 
    and then read out into the `r` register repeatedly.
    """
    x = QuantumRegister(n, "x")
    a = QuantumRegister(n, "a")
    c = QuantumRegister(n, "c")
    e = QuantumRegister(3, "e")
    r = [ClassicalRegister(1) for _ in range(2 * n)]
    circ = QuantumCircuit(x, a, c, e, *r)
    circ.x(x[0])  # start with value 1

    for i in range(2 * n):
        power = 2 * n - 1 - i  # need to handle the highest power first
        circ.h(e[0])
        circ.extend(modulo_mul(n, v ** (1 << power), N))
        for j in range(i):
            circ.p(-pi / 2 ** (i - j), e[0]).c_if(r[j], 1)
        circ.h(e[0])
        circ.measure(e[0], r[i])
        circ.x(e[0]).c_if(r[i], 1)

    return circ


if __name__ == "__main__":
    a, N = 7, (3 * 11)  # working: 2,9: 2,15; 2,21;
    n = ceil(log(N, 2))
    shots = 1
    r = 1

    simulator = Aer.get_backend("qasm_simulator")

    # for j in range(n * 2):
    #     circ = modulo_exp(n, a, j, N)
    #     print(execute(circ, simulator).result().get_counts(circ))

    circ = quantum_period(n, a ** r, N)

    while True:
        # Execute and get counts
        result = execute(circ, simulator, shots=shots).result().get_counts(circ)
        result = {k[::2]: v for k, v in result.items()}

        print(f"Quantum part returned these values: {sorted(result.items())}")

        period = defaultdict(int)
        for value, times in result.items():
            value = int(value, 2)
            period[post_process(value, n, N)] += times

        print(f"New factors of the period: {sorted(period.items())}")

        r *= max(period.keys())
        if a ** r % N == 1:
            print(f"Found the period: {r}")
            factor = factor_finder(r, a, N)
            if factor is not None:
                print(f"Prime factors found: {factor[0]}, {factor[1]}")
            else:
                print("The period didn't result in a factorisation")
            break

        print("Period not found, trying Shor again.")
        if max(period.keys()) != 1:
            circ = quantum_period(n, a ** r, N)
