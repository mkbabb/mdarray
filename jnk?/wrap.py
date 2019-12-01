import numpy as np


def badness(line_length, text_width):
    if (line_length > text_width):
        return float("inf")
    else:
        return (text_width - line_length)**2


def get_breaks(words, text_width):
    N = len(words)
    dp = [0 for i in range(N + 1)]
    breaks = [0] * N

    for i in range(N - 1, -1, -1):
        l = 0
        min_ix = 0
        min_v = float("inf")

        for j in range(i, N):
            l += len(words[j])
            b = badness(l, text_width) + dp[j + 1]

            if (min_v > b):
                min_ix = j
                min_v = b

        breaks[i] = min_ix + 1
        dp[i] = min_v

    return breaks


def get_lines(words, breaks):
    N = len(words)

    i = 0
    line = ""
    while (i < N):
        end = breaks[i]

        for j in range(i, end):
            line += words[j]
            if (j < end):
                line += " "

        line += "\n"
        i = end

    return line
