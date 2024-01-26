def mygen(x, y):
    while x <= y:
        yield x
        x += 1


x=mygen(5, 10)
print(type(x))
for i in x:
    print(i)
