start = int(input('Enter Start Value: '))
stop = int(input('Enter Stop Value: '))
counter = 0
while start <= stop:
    if start % 2 == 0:
        counter += 1

    start += 1
print('Total Even numbers: ', counter)
