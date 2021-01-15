from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit, ClassicalRegister
from qiskit import Aer, execute
from math import pi, log, ceil, gcd
from collections import defaultdict


def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)


def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception("modular inverse does not exist")
    else:
        return x % m


# calculates the overflow bit for a + v if e[1]
# assumes c is zeroed, the result is stored in e[2]
def carry(n: int, v: int) -> QuantumCircuit:
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


# calculates a + v1 if c1 and a + v2 if c2
# assumes c[:-2] is zeroed
def add_const(n: int, v1: int, v2: int) -> QuantumCircuit:
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


# calculates a + v % N if e[1]
# assumes c is zeroed
def modulo_add(n: int, v: int, N: int) -> QuantumCircuit:
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


# calculates v * x % N if e[0]
# assumes a and c are zeroed
def modulo_mul(n: int, v: int, N: int) -> QuantumCircuit:
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
    x = QuantumRegister(n, "x")
    a = QuantumRegister(n, "a")
    c = QuantumRegister(n, "c")
    e = QuantumRegister(3, "e")
    r = [ClassicalRegister(1) for r in range(2 * n)]
    circ = QuantumCircuit(x, a, c, e, *r)
    circ.x(x[0])  # start with value 1

    for i in range(2 * n):
        power = 2 * n - 1 - i  # need to handle the bit power first
        circ.h(e[0])
        circ.extend(modulo_mul(n, v ** (1 << power), N))
        for j in range(i):
            circ.p(-pi / 2 ** (i - j), e[0]).c_if(r[j], 1)
        circ.h(e[0])
        circ.measure(e[0], r[i])
        # circ.x(e[0]).c_if(r[i], 1)

    return circ


def to_expansion(a: int, b: int) -> list:
    if b == 0:
        return []
    return [a // b] + to_expansion(b, a % b)


def from_expansion(e: list) -> (int, int):
    if len(e) == 0:
        return 1, 0
    a, b = from_expansion(e[1:])
    return e[0] * a + b, a


def post_process(y: int, n: int, N: int) -> int:
    Q = 1 << 2 * n
    expansion = to_expansion(y, Q)
    for i in reversed(range(len(expansion))):
        s, r = from_expansion(expansion[: i + 1])
        if r < N and abs(s / r - y / Q) <= 1 / (2 * Q) and r < N:
            return r  # r is very likely to be a factor of the period
    return 0  # not supposed to happen


if __name__ == "__main__":
    a, N = 2, 15
    n = ceil(log(N, 2))
    shots = 5

    circ = quantum_period(n, a, N)
    simulator = Aer.get_backend("qasm_simulator")

    # Execute and get counts
    result = execute(circ, simulator, shots=shots).result().get_counts(circ)
    result = {k[::2]: v for k, v in result.items()}

    print(result)

    period = defaultdict(int)
    for value, times in result.items():
        value = int(value, 2)
        period[post_process(value, n, N)] += times

    print(period)

    for r in list(period.keys()):
        if r % 2:
            continue
        else:
            x = int(a ** (r / 2) % N)
            if (x + 1) % N != 0:
                p, q = gcd(x + 1, N), gcd(x - 1, N)
                if (p * q == N) and p > 1 and q > 1:
                    print("Prime factors found: {}, {}".format(p, q))

    # print(circ.draw("text"))

    # for j in range(10):
    #     circ = modulo_exp(4, 2, j, 15)
    #     simulator = Aer.get_backend("qasm_simulator")
    #     print(execute(circ, simulator).result().get_counts(circ))
