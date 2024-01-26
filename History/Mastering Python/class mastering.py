

class Test:

    grace_score = 10

    def __init__(self):
        self.A = 100
        self.B = 125
        print('Hi')



ob1 = Test()
ob2 = Test()

ob2.grace_score = 25

print(ob1.A, ob1.B, ob1.grace_score)
print(ob2.A, ob2.B, ob2.grace_score)
