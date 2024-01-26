x = int(input('Enter the number of days required by A: '))
y = int(input('Enter the number of days required by B: '))
z = int(input('Enter the number of days required by C: '))

days = (x * y * z)/(x*y + y*z + x*z)

print('Total time taken to complete work if A, B and C work together: %.2f ' % days)
