n = int(input("Enter a number to check whether it is a prime number: "))

if n < 2:
    print(str(n)+" is not a Prime Number")

elif n == 2:
    print(str(n)+" is a Prime Number")

else:
    F = 0
    f = 1
    while f <= n:
        r = n % f
        if r == 0:
            F += 1
        f += 1

    if F == 2:
        print(str(n)+" is a Prime Number")

    else:
        print(str(n) + " is not a Prime Number")
