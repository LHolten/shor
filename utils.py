from math import gcd


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
    # print(f"y {y}, n {n}, N {N}")
    Q = 1 << 2 * n
    expansion = to_expansion(y, Q)
    # print(expansion)
    for i in reversed(range(len(expansion))):
        s, r = from_expansion(expansion[: i + 1])
        # print(f"s {s}, r {r}")
        if r < N and abs(s / r - y / Q) < 1 / (2 * Q):
            # print(f"res {r}")
            return r  # r is very likely to be a factor of the period
    return 1  # not supposed to happen


def factor_finder(r: int, a: int, N: int) -> list:
    if r % 2:
        return None

    x = a ** (r // 2) % N
    if (x + 1) % N != 0:
        p, q = gcd(x + 1, N), gcd(x - 1, N)
        if N % p == 0:
            return [p, N // p]
        if N % q == 0:
            return [q, N // q]

    return None


if __name__ == "__main__":
    # post_process(1115, 6, 33)
    post_process(51, 4, 9)  # supposed to return factor of 6
