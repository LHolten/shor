from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit

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

    for i in range(1, last):
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


# calculates v**e % N
# assumes a and c are zeroed
def modulo_exp(
    circ: QuantumCircuit,
    a: QuantumRegister,
    c: QuantumRegister,
    e: QuantumRegister,
    v: int,
    N: int,
):
    for i in range(len(e)):
        modulo_add(circ, a, c, e[i], v ** i % N, N)


if __name__ == "__main__":
    x = QuantumRegister(4, "x")
    a = QuantumRegister(4, "a")
    c = QuantumRegister(4, "c")
    circ = QuantumCircuit(x, a, c)
    modulo_exp(circ, a, c, x, 2, 15)
    print(circ.draw("text"))
