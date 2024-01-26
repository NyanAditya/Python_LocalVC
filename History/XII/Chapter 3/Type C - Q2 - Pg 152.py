# WAF to calculate the volume of a box

def vol(l=0.0, w=0.0, h=0.0):
    return l * w * h


length = float(input('Enter Length: '))
width = float(input('Enter width: '))
height = float(input('Enter height: '))

print('Volume of the box is ', vol(length, width, height))
