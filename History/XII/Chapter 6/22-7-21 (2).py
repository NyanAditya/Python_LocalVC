# WAF to reverse a string

def rev(x):
    ind = len(x)-1
    print(x[ind], end='')
    if ind > 0:
        return rev(x[:ind])

    else:
        return ''


name = input('Enter Name: ')
rev(name)
