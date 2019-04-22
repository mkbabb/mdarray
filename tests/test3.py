from __future__ import annotations

import operator
import random
import timeit
import typing
from functools import reduce
import random

import numpy as np

from core import *


def get_ret_shaped2(buff, arr, new_shape, axis, keepdims):
    buff = tomdarray(buff)
    if buff.size > 1:
        if not keepdims:
            new_shape.pop(axis)
            new_shape = buff.shape + new_shape
    else:
        if keepdims:
            new_shape[axis] = 1
        else:
            new_shape.pop(axis)
    arr_out = zeros(new_shape)
    return arr_out


def _insert_into_flattened2(buff, arr_out, j):
    if isinstance(buff, ma.multiArray):
        buff = buff.data

    if isinstance(buff, list):
        for n, i in enumerate(buff):
            arr_out.data[n + j] = i
        j += len(buff)
    else:
        arr_out.data[j] = buff
        j += 1
    return j


def reduce_iter(arr: ma.multiArray, faxis: int,
                func: typing.Callable[[list], list],
                keepdims: bool = False) -> ma.multiArray:
    mdim = arr.mdim
    roll_axis(arr, faxis)
    mditer = MDIter(arr)
    shape = arr.shape
    new_shape = list(shape)
    buff = [0] * shape[0]

    j = 0
    for i in range(shape[faxis]):
        fbuff = mditer.grapple(buff, 1, func)
        if i == 0:
            arr_out = get_ret_shaped2(fbuff, arr, new_shape,
                                      faxis, keepdims)
        j = _insert_into_flattened2(fbuff, arr_out, j)

    roll_axis(arr, faxis, mdim - 1)
    return arr_out


def lcs(s1, s2):
    N, M = len(s1), len(s2)
    arr = [[None for i in range(M + 1)] for j in range(N + 1)]

    def recurse(n, m):
        if arr[n][m] != None:
            return arr[n][m]
        elif n == 0 or m == 0:
            seq = ""
        elif s1[n - 1] == s2[m - 1]:
            seq = recurse(n - 1, m - 1)
            seq += s1[n - 1]
        elif s1[n - 1] != s2[m - 1]:
            tmp1 = recurse(n - 1, m)
            tmp2 = recurse(n, m - 1)
            seq = max(tmp1, tmp2, key=len)
        else:
            seq = ""
        arr[n][m] = seq
        return seq

    seq = recurse(N, M)
    return seq


def lccs(s1, s2):
    N, M = len(s1), len(s2)
    arr = [[None for i in range(M + 1)] for j in range(N + 1)]

    def recurse(n, m, seq):
        if arr[n][m] != None:
            return seq
        elif n == 0 or m == 0:
            return seq
        elif s1[n - 1] == s2[m - 1]:
            seq = recurse(n - 1, m - 1, s1[n - 1] + seq)
        elif s1[n - 1] != s2[m - 1]:
            tmp1 = recurse(n - 1, m, "")
            tmp2 = recurse(n, m - 1, "")
            seq = max(seq, tmp1, tmp2, key=len)
            arr[n][m] = seq
        return seq

    seq = recurse(N, M, "")
    return seq


def sort_lccs(seq):
    N = len(seq)
    seq = [[i, False] for i in seq]
    seqs = []

    for i in range(N):
        seqs_i = [seq[i][0]]
        seq_i = seq[i][0]
        for j in range(i + 1, N):
            seq_ij = seq[j][0]
            if (not seq[i][1] and not seq[j][1] and
                    (len(seq_i) / len(seq_ij) >= 0.5)):
                sub_seq_i = lccs(seq_i, seq_ij)
                thresh = (len(sub_seq_i) / len(seq_i) >= 0.5
                          and len(sub_seq_i) / len(seq_ij) >= 0.5)
                if thresh:
                    if len(seqs_i) == 1:
                        seq[j][1] = True
                        seqs_i.append(seq_ij)
                    else:
                        group = True
                        for k in seqs_i:
                            sub_seq_k = lccs(seq_ij, k)
                            thresh = (len(sub_seq_k) / len(seq_ij) >= 0.5
                                      and len(sub_seq_k) / len(k) >= 0.5)
                            if not thresh:
                                group = False
                                break
                        if group:
                            seqs_i.append(seq_ij)
                            seq[j][1] = True
        if len(seqs_i) > 1:
            seq[i][1] = True
            seqs_i = sorted(seqs_i)
            seqs.append(seqs_i)
    seqs = flatten_list(seqs, -1)[0] + sorted([i[0] for i in seq if not i[1]])
    return seqs


random.seed(1)
words = ["log", "frog", "dog", "hog", "fick", "slick", "tick", "rays", "hays", "mays"
         "fray", "lay", "gay", "hag", "bag", "tag", "rag", "lag", "zealot", "ameliorate",
         "buck", "cluck", "duck", "chuck", "muck", "tuck", "obsequeious", "licentious", "pretentious",
         "cog", "cam", "deviantjam", "ham", "biscuit", "triscuit", "play", "gorge", "lorge", "hello", "yellow",
         "cookie", "bookie", "book", "have", "to", "be", "cookin", "by", "the", "book", "my", "word", "what", "does",
         "that", "mean", "my", "men", "hen", "den", "quest", "jest", "zest", "for", "lime"]

# words = ["ham", "cam", "jam", "biscuit", "triscuit", "bug", "rug", "hug"]
words = ["dog", "log", "hen", "ham", "jam"]
random.shuffle(words)
print(words)

MAX_WORDS = 5
MAX_WORD_LEN = 5


def get_some_words(words):
    _words = []
    N = len(words)

    for i in range(MAX_WORDS // 2):
        word_i = words[i]
        word_Ni = words[N - (i + 1)]
        if len(word_i) <= MAX_WORD_LEN:
            _words.insert(0, word_i)
        if len(word_Ni) <= MAX_WORD_LEN:
            _words.append(word_Ni)

    if N > MAX_WORDS:
        _words.insert(MAX_WORDS // 2, "...")

    return ", ".join(_words)


# s = get_some_words(words)
# print(s)


# s1 = "ham"
# s2 = "cam"
# seq = lccs(s2, s1)
# print(seq)
sorted_words = sort_lccs(words)
print(sorted_words)
