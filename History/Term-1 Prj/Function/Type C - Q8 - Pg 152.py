# WAF that takes two numbers and returns the number that has minimum one's digit

def ones(num1, num2):
    if int(str(num1)[-1]) > int(str(num2)[-1]):
        return num2

    else:
        return num1


x = int(input('Enter 1st Number: '))
y = int(input('Enter 2nd Number: '))
print(ones(x, y))
