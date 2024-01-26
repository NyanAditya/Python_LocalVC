num = int(input('Enter a Number: '))

for i in range(2, num):
    if num % i == 0:
        print("The number is Not Prime")
        break
else:
    print('The Number is Prime')
