n=int(input("Enter a number to check whether it is a Prime Number:"))

if n<2:
     print(str(n)+" is Not a Prime Number")

elif n==2:
     print(str(n)+" is a Prime Number")

else:
     c=0
     i=1
     while i<=n:
          f=n%i
          if f==0:
               c+=1
          i+=1

     if c==2:
          print(str(n)+" is a Prime Number")
     else:
          print(str(n)+" is Not a Prime Number")

          

