num = int(input("Enter the no."))
fac, fac2, rev = 0, 0, 0
tmp_num = num
div_num = 1

while div_num <= num:
    if num % div_num == 0:
        fac += 1
    div_num += 1
div_num = 1
while tmp_num > 0:
    dig = num // 10
    rev = rev * 10 + dig
while div_num <= rev:
    if rev % div_num == 0:
        fac2 += 1
div_num += 1

if fac == fac2 == 2:
    print("The given no. is a twisted prime no.")
else:
    print("The given no. is not a twisted prime no.")
