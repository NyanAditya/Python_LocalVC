num1 = int(input("Enter 1st Number: "))
num2 = int(input("Enter 2nd Number: "))

MAX = max(num1, num2)

while MAX >= 1:
	if num1 % MAX == 0 and num2 % MAX == 0:
		print(MAX)
		break

	MAX -= 1
