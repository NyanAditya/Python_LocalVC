num = int(input("Enter a Binary Number: "))
power = len(str(num)) - 1
s = 0

for i in str(num):
	decimal = int(i) * (2**power)
	s = s * 10 + decimal
	power -= 1

print(s)
