f = None
for i in range(5):
    f = open('data.txt', 'w')
    if i > 2:
        break
print(f.closed)
