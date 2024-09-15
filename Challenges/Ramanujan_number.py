"""
A Ramanujan number is a number that can be expressed as the sum of two cubes in two different ways.

Example:
Input: 1729
Output: The sum of two cubes that equals 1729 can be expressed as 1^3 + 12^3 or 9^3 + 10^3.
"""


def isRamanujan(num: int) -> str:
    
    Ramanujan_cube_pairs: list[(int, int)] = []
    
    i = j = 1
    
    while i**3 <= num:
        
        j = 1
    
        while (i**3 + j**3) <= num:
            
            if (i**3 + j**3) == num:
                Ramanujan_cube_pairs.append((j, i))
            
            if len(Ramanujan_cube_pairs) == 2:
                return f"Yes, {num} = {Ramanujan_cube_pairs[0][0]}³ + {Ramanujan_cube_pairs[0][1]}³ = {Ramanujan_cube_pairs[1][0]}³ + {Ramanujan_cube_pairs[1][1]}³"
            
            j += 1
            
        i += 1
            
    return "No!"
        
        
num: int = int(input("Enter a Number: "))
print(isRamanujan(num))
