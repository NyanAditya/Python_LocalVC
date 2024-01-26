num = int(input("Enter a Number: "))
binary = ''
while num > 0:
	dig = num % 2
	binary = str(dig) + binary
	num //= 2

print(binary)
