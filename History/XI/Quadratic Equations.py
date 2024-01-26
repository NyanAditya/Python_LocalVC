# Python3 program to find upstream
# and downstream speeds of a boat.

# Function to calculate the
# speed of boat downstream
def Downstream(b, s):
	return (b + s)


# Function to calculate the
# speed of boat upstream
def Upstream(b, s):
	return (b - s)


# Driver Code

# Speed of the boat in still water(B)
# and speed of the stream (S) in km/hr
B = 10;
S = 4
print("Speed Downstream = ", Downstream(B, S), " km/hr",
      "\nSpeed Upstream = ", Upstream(B, S), " km/hr")

# This code is contributed by Anant Agarwal.
