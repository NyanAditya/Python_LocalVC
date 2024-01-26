def Findoutput():
    L = "earn"
    X = " "
    L1 = []
    count = 1
    for i in L:
        if i in ['a', 'e', 'i', 'o', 'u']:
            X = X + i.swapcase()
        else:
            if count % 2 != 0:
                X = X + str(len(L[: count]))
            else:
                X = X + i
        count = count + 1
    print(X)


Findoutput()
