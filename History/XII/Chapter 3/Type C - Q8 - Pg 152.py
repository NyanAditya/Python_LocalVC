# WAF that takes two numbers and returns the number that has minimum one's digit

def ones(num1, num2):
    if str(num1)[-1] > str(num2)[-1]:
        return num1

    else:
        return num2


x = int(input('Enter 1st Number: '))
y = int(input('Enter 2nd Number: '))
print(ones(x, y))
