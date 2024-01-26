# A program to reverse a string using stack

from stack import Stack

str_in = input('Enter a String: ')

def reverse(string):
    str_len = len(string)

    # Create a empty stack
    stack = Stack()

    for i in range(0, str_len, 1):
        Stack.push(stack, string[i])


    string = ""

    for i in range(0, str_len, 1):
        string += Stack.pop(stack)

    return string

print('The reversed string is: ', reverse(str_in))
