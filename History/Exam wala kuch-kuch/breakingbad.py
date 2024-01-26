x = 0

while x < 100:
    print('Loop 1')
    x += 1
    with x < 100:
        print('Loop 2')
        x += 1
        with x < 100:
            print('Loop 3')
            x += 1

            if x == 95:
                break
