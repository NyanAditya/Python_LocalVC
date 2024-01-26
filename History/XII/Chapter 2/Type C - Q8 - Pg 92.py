a, b = 0, 1
l = [0, 1]

for i in range(7):
    a, b = b, a + b
    l.append(b)

fib = tuple(l)
print(fib)
