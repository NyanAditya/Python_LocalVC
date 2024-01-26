def Change(p, q=30):
    p = p + q
    q = p - q
    print(p, '#', q)
    return (p)


r = 150
s = 100
r = Change(r, s)
print(r, '#', s)
s = Change(s)
