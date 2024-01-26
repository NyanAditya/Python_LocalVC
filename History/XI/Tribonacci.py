a, b, c = 0, 1, 1
print(a, b, c, sep='\n')

while a+b+c <= 100:
    nxt_num = a + b + c
    print(nxt_num)
    a, b, c = b, c, a + b + c  # assigning the value of b to a before it changes
