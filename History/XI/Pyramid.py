n = int(input("Enter the Step Size:"))
for i in range(1, n + 1):
	k = i
	for j in range(1, i + 1):
		print('*', end='')
		k += 1
		print()
