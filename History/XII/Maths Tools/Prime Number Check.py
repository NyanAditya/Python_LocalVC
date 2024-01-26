num = int(input('Prime Number Check: '))
factors = 0

if num < 1:
    print('Prime Number are +ve! Damn!!')

elif num == 1:
    print('1 is Special ^_^')

else:
    for i in range(2, num + 1):
        if num % i == 0:
            factors += 1

    if factors == 1:
        print('Its PRIME!')

    else:
        print('Its NOT prime!')
