a, b = 0, 1,
print(a, b, sep='\n')

while a+b <= 100:
    nxt_num = a + b
    print(nxt_num)
    a, b = b, a+b  # assigning the value of b to a before it changes
