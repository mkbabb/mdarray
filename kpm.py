import numpy as np
import random
from typing import Dict, List
import datetime
import timeit


def boyer_moore_3(string, pat):
    N = len(string)
    M = len(pat)

    delta = {}

    for i in range(M - 1):
        delta[pat[i]] = M - (i + 1)

    matches = []

    i = M - 1
    while (i < N):
        pos_i = i
        for j in range(M - 1, -1, -1):
            if string[i] != pat[j]:
                if pat[j] not in delta:
                    i = pos_i + M
                else:
                    i = pos_i + delta[pat[j]]
                break
            else:
                i -= 1

            if (j == 0):
                matches.append(i + 1)
                i += M + 1
    return matches


def _boyer_moore_2(string, pat, delta, start):
    N = len(string)
    M = len(pat)

    i = start
    while i < N - M + 1:
        j = M - 1

        while (string[i + j] == pat[j]):
            j -= 1

            if (j < 0):
                return i

        c = string[i + M - 1]
        i += M - 1

        if (c in delta):
            i -= delta[c]
        else:
            i += 1

    return -1


def boyer_moore_2(string, pat):
    delta = {}

    M = len(pat)

    for i in range(M - 1):
        delta[pat[i]] = i

    matches = []
    start = 0

    while (True):
        start = _boyer_moore_2(string, pat, delta, start)

        if (start == -1):
            break
        else:
            matches.append(start)
            start += M + 1

    return matches


def _boyer_moore(string, pat, delta, start):
    N = len(string)
    M = len(pat)
    i = start

    while i < N - M + 1:
        j = M - 1

        while (string[i + j] == pat[j]):
            j -= 1
            if (j < 0):
                return i

        c = string[i + M - 1]
        if c not in delta:
            i += M
        else:
            i += delta[c]
    return -1


def boyer_moore(string, pat):
    delta = {}

    M = len(pat)

    for i in range(M - 1):
        delta[pat[i]] = M - (i + 1)

    matches = []
    start = 0

    while (True):
        start = _boyer_moore(string, pat, delta, start)

        if (start == -1):
            break
        else:
            matches.append(start)
            start += M + 1

    return matches


def find_all(string, pat):
    matches = []
    start = 0

    while (True):
        start = string.find(pat, start)
        if (start == -1):
            break
        else:
            matches.append(start)
            start += len(pat) + 1

    return matches


# for i in range(1000):
#     string = "".join(random.choices([i for i in ALPHABET], k=50000))
#     N = len(string)
#     a = random.randint(0, N)
#     b = random.randint(0, N)

#     while a == b or a > b:
#         a = random.randint(0, N)
#         b = random.randint(0, N)

#     pat = string[a:b]

#     poss = boyer_moore(string, pat)
#     boss = find_all(string, pat)

#     for i in range(len(poss)):
#         assert(poss[i] == boss[i])

#     if (poss[0] != p):
#         for pos in poss:
#             print(string[pos: pos + len(pat)])

# string = "storethatisapplethatisapplesapple"

# pat = "apple"

# poss = boyer_moore(string, pat)
# for pos in poss:
#     print(string[pos:pos + len(pat)])
