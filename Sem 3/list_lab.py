user_input = []

for _ in range(10):
    x = int(input(f"Enter Element {_+1}"))
    user_input.append(x)
    
user_input.sort()

print(f"Largest: {user_input[-1]}")

