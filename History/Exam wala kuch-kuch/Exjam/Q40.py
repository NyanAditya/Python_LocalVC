def fxn():
    f = open('IMP')
    e = 0
    u = 0
    while True:
        l = f.readline()
        if not l:
            break
        for i in l:
            if i == 'E' or i == 'e':
                e = e + 1
            elif i == 'U' or i == 'u':
                u = u + 1
    print(e)
    print(u)
    f.close()


fxn()
