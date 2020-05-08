import timeit


stmt = '''
import random
random.seed(4)
ALPHABET = "abcdefghigklmnopqrstuvwxyz"

for i in range(100):
    string = "".join(random.choices([i for i in ALPHABET], k=500000))
    N = len(string)
    a = random.randint(0, N)
    b = random.randint(0, N)

    while a == b or a > b:
        a = random.randint(0, N)
        b = random.randint(0, N)

    pat = string[a:b]

    poss = f(string, pat)
'''

count = 10

t1 = timeit.timeit(stmt=stmt,
                   setup='''
from kpm import find_all as f
''', number=count)

t2 = timeit.timeit(stmt=stmt,
                   setup='''
from kpm import boyer_moore as f
''', number=count)

t3 = timeit.timeit(stmt=stmt,
                   setup='''
from kpm import boyer_moore_2 as f
''', number=count)

t4 = timeit.timeit(stmt=stmt,
                   setup='''
from kpm import boyer_moore_3 as f
''', number=count)

print(t1, t2, t3, t4)
