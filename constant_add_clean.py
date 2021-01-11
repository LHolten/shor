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
def modulo_add(n: int, v: int, N: int,) -> QuantumCircuit:
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
def modulo_mul(n: int, v: int, N: int,) -> QuantumCircuit:
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


if __name__ == "__main__":
    x = QuantumRegister(4, "x")
    a = QuantumRegister(4, "a")
    c = QuantumRegister(4, "c")
    e = QuantumRegister(3, "e")
    r = ClassicalRegister(4, "r")
    circ = QuantumCircuit(x, a, c, e, r)

    circ.x(e[0])  # enable multiplication
    circ.x(x[0])  # start with value 1
    circ.extend(modulo_mul(4, 7, 15))
    circ.extend(modulo_mul(4, 7, 15))
    circ.measure(x, r)

    # Select the StatevectorSimulator from the Aer provider
    simulator = Aer.get_backend("qasm_simulator")

    # Execute and get counts
    result = execute(circ, simulator, shots=10).result()
    print(result.get_counts(circ))

    # print(circ.draw("text"))
