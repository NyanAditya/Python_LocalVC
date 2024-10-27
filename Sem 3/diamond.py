# WAP to print a Diamond shape in python

def Shine_like_a_Diamond(rows):
    n = 0
    for i in range(1, rows + 1):
        # loop to print spaces
        for j in range(1, (rows - i) + 1):
            print(end=" ")

        # loop to print star
        while n != (i):
            print("* ", end="")
            n = n + 1
        n = 0
        print()

    k = 1
    n = 1
    for i in range(1, rows):
        # loop to print spaces
        for j in range(1, k + 1):
            print(end=" ")
        k = k + 1

        # loop to print star
        while n <= ((rows - i)):
            print("* ", end="")
            n = n + 1
        n = 1
        print()


rows = int(input("Enter Rows: "))
Shine_like_a_Diamond(rows)
