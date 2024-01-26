a = int(input("Enter a Number to check whether it is a prime number\n"))

if a < 2:
	print("not prime")

elif a == 2:
	print("prime")

else:
	c = 1
	d = 0
	while c <= a:
		r = a % c
		if r == 0:
			d += 1
			f = [c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c, c]
		c += 1
	if d == 2:
		print("Prime")
	else:
		print("Not Prime")

print(f)
