members = ["ab", "bc", "cd", "de", "ef", "fg", "ac", "ce", "eg", "bd",
           "df", "nm", "ml", "lk", "kj", "ji", "nl", "lj", "jh", "mk", "ki",
           "an", "cl", "ej", "gh", "bm", "dk", "fi", "ih"]
n_members = []

abc = "abcdefghijklmnopqrstuvwxyz"
abc = "".join(sorted([i for i in abc]))

d = {i: str(n + 1) for n, i in enumerate(abc)}
for i in members:
    tmp = []
    for j in i:
        tmp.append(d[j])
    n_members.append(",".join(tmp))

t = ";".join(n_members)
print(t)
