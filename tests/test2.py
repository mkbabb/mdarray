import core


# words = ["woon", "swoon", "june", "elephant",
#          "bazaar", "billeted", "obsequious"]

words = ["onword", "ameliorate"]

def min_letters(seq):
    N = len(seq)
    d = {}
    for i in range(N):
        seq_i = seq[i]
        N_i = len(seq_i)
        d_i = {}

        for j in range(N_i):
            seq_ij = seq_i[j]

            if seq_ij not in d_i:
                d_i[seq_ij] = 1
            else:
                d_i[seq_ij] += 1
        
        for key, value in d_i.items():
            if key not in d:
                d[key] = value
            else:
                if d[key] < value:
                    d[key] = value

    return d


d = min_letters(words)
for i in d.items():
    print(i)
