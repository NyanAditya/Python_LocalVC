num1 = int(input("Enter First Number: "))
num2 = int(input("Enter Second Number: "))

MAX = max(num1, num2)

while MAX <= (num1 * num2):
	if MAX % num1 == 0 and MAX % num2 == 0:
		print(MAX)
		break

	MAX += 1
