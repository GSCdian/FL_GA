a = []

for i in range(10):
    a.append(i)
    if len(a) > 1:
        a[-1] += a[-2]

print(a)