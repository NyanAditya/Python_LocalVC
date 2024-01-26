with open('Phone.txt', mode='r') as F:
    for rec in F.readlines():
        print(rec, end='')

F.close()

