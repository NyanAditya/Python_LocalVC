users = []

for i in range(1, 11):
    x = input('Enter User {} Name: '.format(i))
    users.append(x)


def big_name(users):
    for name in users:
        if len(name) > 5:
            print(name)


big_name(users)
