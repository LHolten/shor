"""
Created on Mon Jan 11 14:13:29 2021

@author: https://stackoverflow.com/questions/15347174/python-finding-prime-factors
"""

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
print(prime_factors(524287*2147483647))
