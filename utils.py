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
