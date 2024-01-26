num = int(input('Enter a Number: '))
tmp_var = num
rev = counter = 0
while num > 0:
    rev = rev*10 + num % 10
    num //= 10

if tmp_var == rev:
    num = tmp_var
    while num > 0:
        if tmp_var % num == 0:
            counter += 1

        num -= 1

    if counter == 2:
        print('Pal-Prime Number')
    else:
        print('Palindrome Number')

else:
    print('Not a palindrome number')
