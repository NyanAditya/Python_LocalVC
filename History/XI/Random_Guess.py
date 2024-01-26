from random import randint
sn = randint(1,100)
un = -1
tc = 1

while sn != un:
     print ("%d try: " % tc, end='')
     un = int(input())

     if un<sn:
          print("Too less")

     elif un>sn:
          print("Too much")

     else:
          print("You gussed it!")

     tc += 1
     
