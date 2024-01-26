# Program to display the Fibonacci sequence up to n-th term
terms = int(input("Number of Terms: "))
# first two terms
n1, n2 = 0, 1
count = 0
# check if the number of terms is valid
if terms <= 0:
	print("Please enter a positive integer")
elif terms == 1:
	print("Fibonacci sequence up to", terms, ":")
	print(n1)
else:
	print("Fibonacci sequence:")
	while count < terms:
		print(n1, end="\n")
		nth = n1 + n2
		# update values
		n1 = n2
		n2 = nth
		count += 1
