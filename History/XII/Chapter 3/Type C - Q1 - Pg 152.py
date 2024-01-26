def moneyfxn(amount):
    return amount * 73.96


USD = float(input('Enter Amount: '))

print('${} equals Rs.{}'.format(USD, moneyfxn(USD)))
