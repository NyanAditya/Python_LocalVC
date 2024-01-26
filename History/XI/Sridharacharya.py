from math import sqrt
print("General Quadratic Formula: ax^2 + bx + c\n")

a = int(input("Enter Coefficient of a: "))
b = int(input("Enter Coefficient of b: "))
c = int(input("Enter Constant Term: "))

D = (b * b) - 4 * a * c

alpha = (-b + sqrt(D)) / (2 * a)
beta = (-b - sqrt(D)) / (2 * a)

if D > 0:
	print("Real Roots are:\n")
	print(alpha, "and", beta)

elif D == 0:
	print("Equal Roots are:\n")
	print(alpha, "and", beta)

else:
	print("No Real Roots for this equation!!")
# This is a Comment
a = 4 is 5
