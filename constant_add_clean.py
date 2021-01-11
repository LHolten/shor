from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit, ClassicalRegister
from qiskit import Aer, execute


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


# calculates the overflow bit for a + v while leaving everything else untouched
# assumes c is zeroed, the result is stored in c[-1]
def carry(circ: QuantumCircuit, a: QuantumRegister, c: QuantumRegister, v: int):
    assert len(a) == len(c)
    last = len(a) - 1

    if v & 1:
        circ.cx(a[0], c[0])

    for i in range(1, len(a)):
        if v & 1 << i:
            circ.cx(a[i], c[i])
            circ.x(a[i])
        circ.ccx(c[i - 1], a[i], c[i])

    if v & 1 << last:
        circ.x(a[last])

    for i in reversed(range(1, last)):
        circ.ccx(c[i - 1], a[i], c[i])
        if v & 1 << i:
            circ.x(a[i])
            circ.cx(a[i], c[i])

    if v & 1:
        circ.cx(a[0], c[0])


# calculates a + v1 if c1 and a + v2 if c2
# assumes c[:-2] is zeroed
def add_const(
    circ: QuantumCircuit, a: QuantumRegister, c: QuantumRegister, v1: int, v2: int
):
    assert len(a) == len(c)
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


# calculates a + v % N if i
# assumes c is zeroed
def modulo_add(
    circ: QuantumCircuit,
    a: QuantumRegister,
    c: QuantumRegister,
    i: Qubit,
    v: int,
    N: int,
):
    carry(circ, a, c, v - N)  # compare v - N < a
    circ.ccx(i, c[-1], c[-2])
    circ.cx(c[-2], i)
    add_const(circ, a, c[:-1] + [i], v, v - N)
    circ.cx(c[-2], i)
    circ.ccx(i, c[-1], c[-2])
    carry(circ, a, c, v)  # compare v < a


# calculates a - v % N if i
# assumes c is zeroed
def modulo_sub(
    circ: QuantumCircuit,
    a: QuantumRegister,
    c: QuantumRegister,
    i: Qubit,
    v: int,
    N: int,
):
    carry(circ, a, c, v)  # compare v < a
    circ.ccx(i, c[-1], c[-2])
    circ.cx(c[-2], i)
    add_const(circ, a, c[:-1] + [i], -v, N - v)
    circ.cx(c[-2], i)
    circ.ccx(i, c[-1], c[-2])
    carry(circ, a, c, v - N)  # compare v - N < a


# calculates v * x % N
# assumes a and c are zeroed
def modulo_mul(
    circ: QuantumCircuit,
    a: QuantumRegister,  # sum
    c: QuantumRegister,  # zero
    x: QuantumRegister,  # mul
    e: QuantumRegister,  # control bit
    v: int,
    N: int,
):
    for i in range(len(x)):
        circ.ccx(e[0], x[i], e[-1])
        modulo_add(circ, a, c, e[-1], v << i % N, N)
        circ.ccx(e[0], x[i], e[-1])
    for i in range(len(x)):
        circ.cswap(e[0], a[i], x[i])
    for i in range(len(x)):
        circ.ccx(e[0], x[i], e[-1])
        modulo_sub(
            circ, a, c, e[-1], modinv(v << i, N), N
        )  # need to calculate modular multiplicative inverse somehow
        circ.ccx(e[0], x[i], e[-1])


def modulo_exp(
    circ: QuantumCircuit,
    a: QuantumRegister,  # sum
    c: QuantumRegister,  # zero bits
    x: QuantumRegister,  # mul
    e: QuantumRegister,  # exponent
    v: int,
    N: int,
):
    circ.x(x[0])
    for i in range(len(a)):
        if i != 0:
            circ.swap(e[0], e[i])
        modulo_mul(circ, a, c, x, e, v ** (2 << i) % N, N)
        if i != 0:
            circ.swap(e[0], e[i])


if __name__ == "__main__":
    x = QuantumRegister(4, "x")
    a = QuantumRegister(4, "a")
    c = QuantumRegister(4, "c")
    e = QuantumRegister(5, "e")
    r = ClassicalRegister(4, "r")
    circ = QuantumCircuit(x, a, c, e, r)

    # modulo_exp(circ, a, c, x, e, 2, 15)
    circ.x(c[-1])
    add_const(circ, a, c, 13, 5)
    circ.x(c[-1])
    carry(circ, a, c, 11)
    circ.measure(c, r)

    # Select the StatevectorSimulator from the Aer provider
    simulator = Aer.get_backend("qasm_simulator")

    # Execute and get counts
    result = execute(circ, simulator, shots=10).result()
    print(result.get_counts(circ))

    # print(circ.draw("text"))
