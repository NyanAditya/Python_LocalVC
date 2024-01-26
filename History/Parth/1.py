def check_stack_isEmpty(stk):
    if stk==[]:
        return True
    else:
        return False
    # An empty list to store stack elements, initially empty
    s=[]
    top = None # This is top pointer for push and pop
def main_menu():
    while True:
        print("Stack Implementation")
        print("1 - Push")
        print("2 - Pop")
        print("3 - Peek")
        print("4 - Display")
        print("5 - Exit")
        ch = int(input("Enter the your choice:"))
        if ch==1:
            el = int(input("Enter the value to push an element:"))
            push(s,el)
        elif ch==2:
            e=pop_stack(s)
            if e=="UnderFlow":
                print("Stack is underflow!")
            else:
                print("Element popped:",e)
        elif ch==3:
            e=pop_stack(s)
            if e=="UnderFlow":
                print("Stack is underflow!")
            else:
            print("The element on top is:",e)
        elif ch==4:
