exp = 1

while pow(2, exp) <= 20:
    output = pow(-1, exp+1)*pow(2, exp)
    print(output)
    exp += 1
