# A python program to perform various operations on stack using Stack class.
from stack import Stack

STK = Stack()

choice = 0
while choice < 5:
    print('Stack Operations')
    print('1 - Push element')
    print('2 - Pop element')
    print('3 - Peep element')
    print('4 - Search for the element')
    print('5 exit')
    choice = int(input('Your choice: '))

    if choice == 1:
        element = int(input('Enter element: '))
        STK.push(element)

    elif choice == 2:
        element = STK.pop()
        if element == -1:
            print('The stack is empty!')
        else:
            print('Popped element= ', element)

    elif choice == 3:
        element = STK.peep()
        print('Topmost element= ', element)

    elif choice == 4:
        element = int(input('Enter element: '))
        pos = STK.search(element)
        if pos == -1:
            print('The Stack is empty!')

        elif pos == -2:
            print('Element not found in the Stack')

        else:
            print('Element found at position: ', pos)

    else:
        break

    print('Stack=', STK.display())
