import pickle


def display(eno):
    f = open("Employee.dat", "rb")
    totSum = 0
    try:
        while True:
            R = pickle.load(f)
            if type(R[0]) == type(str()):
                continue
            elif R[0] == eno:
                continue

            totSum = totSum + R[2]

    except EOFError:
        f.close()
    print(totSum)


display(103)
