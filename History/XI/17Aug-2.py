random_list = [1, 7, 9, 14, 18, 70, 1400, 280]
counter = 0

for i in random_list:
    if i % 7 == 0:
        counter += 1

print(random_list)
print('Multiples of 7 in the given list are ', counter)
