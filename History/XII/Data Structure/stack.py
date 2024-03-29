# A python program to create a class that
# can perform fundamental operations of a STACK

class Stack:
    def __init__(self):
        self.st = []

    def isempty(self):
        return self.st == []

    def push(self, element):
        self.st.append(element)

    def pop(self):
        if self.isempty():
            return -1

        else:
            return self.st.pop()

    def peep(self):
        n = len(self.st)
        return self.st[n-1]

    def display(self):
        return self.st

    def search(self, element):
        if self.isempty():
            return -1

        else:
            try:
                n = self.st.index(element)
                return len(self.st)-n
            except ValueError:
                return -2

