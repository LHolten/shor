# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:22:34 2021

@author: Evanl
"""

import matplotlib.pyplot as plt
import numpy as np
from math import gcd
from scipy.signal import find_peaks
import time


def fft_primefactor(a,num_it,N):
    for i in range(num_it):
        #print(i)
        #print(a)
        while gcd(a,N) != 1:
            a+=1
            #print(a)
        x = np.arange(20,dtype=object)
        fx = a**x % N
        freq = np.fft.rfftfreq(len(x))[1:-1]
        fft = np.abs(np.fft.rfft(fx, norm="ortho")[1:-1])
        r = np.round(1/(freq[(np.max(fft)==fft)]))
        if not isinstance(r, int): 
            r=r[0]
        
        if (r % 2):
            a+=1    
        else:
            x = int(a**(r/2) % N)
            if (x + 1) % N != 0:
                p,q = gcd(x+1,N), gcd(x-1,N)
                if (p*q==N) and ((p and q) >1):
                    return [p,q]
                elif p > 1:
                    other = N/p
                    return [other,p]
                elif q > 1:
                    other = N/q
                    return [other,q]
                else:
                    a+=1
            else:
                a+=1
        if i == num_it-1:
            print("ran out of iterations")
            return

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


N=33
a=2
iterations = 10

"""
time1=time.time()
print(fft_primefactor(a,iterations,N))
time2=time.time()
time3=time.time()
print(prime_factors(N))
time4 = time.time()
print(time2-time1)
print(time4-time3)
"""



#BENCHMARK
time1=time.time()
for i in range(10):
    #print(fft_primefactor(a,iterations,N))
    fft_primefactor(a,iterations,N)
time2=time.time()
time3=time.time()
for i in range(10):
    #print(prime_factors(N))
    prime_factors(N)
time4 = time.time()
print(time2-time1)
print(time4-time3)


