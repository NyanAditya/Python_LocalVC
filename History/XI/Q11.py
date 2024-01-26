P = float(input('Enter the principal: '))
R = float(input('Enter the rate of interest per annum: '))
T = float(input('Enter the time in years: '))

SI = (P * R * T) / 100

amount = SI + P

print('Total amount:', amount)
