import pickle

try:
    with open('Employee.dat', 'rb') as fr:
        while True:
            print(pickle.load(fr))

except EOFError:
    pass
