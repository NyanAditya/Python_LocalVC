s = 0

for i in range(1, 10 + 1):
	if i == 1:
		print("")

	else:
		print(" + ", end="")

	print(i, end="")
	s = s + i

print(" = ", s)
