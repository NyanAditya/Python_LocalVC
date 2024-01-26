# Python code to demonstrait
# nonlocal keyword

# Nested function to demonstrait
# nonlocal keyword
def geek_func():
    # local variable to geek_func
    geek_name = "geekforgeeks"

    # First Inner function
    def geek_func1():
        geek_name = "GeekforGeeks"

        # Second Inner function
        def geek_func2():
            # Declairing nonlocal variable
            nonlocal geek_name
            geek_name = 'GEEKSFORGEEKS'

            # Printing our nonlocal variable
            print(geek_name)

        # Calling Second inner function
        geek_func2()

    # Calling First inner function
    geek_func1()

    # Printing local variable to geek_func
    print(geek_name)


geek_func()
