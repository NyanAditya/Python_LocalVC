# bool datatype coercion | conversion

f = False  # boolean False
t = True   # boolean True

print(int(t), int(f), sep=' और ')          # integers
print(float(t), float(f), sep=' और ')      # float
print(complex(t), complex(f), sep=' और ')  # complex
print(str(t), str(f), sep=' और ')          # strings
print(bytes(t), bytes(f), sep=' और ')      # bytes
print(range(t), range(f), sep=' और ')      # range

print()

print(bool(''), bool(' '), sep=' और ')          # strings
print(bool('False'), bool('True'), sep=' और ')  # strings 2.0
print(bool([]), bool([1]), sep=' और ')          # list
print(bool({}), bool({'': ''}), sep=' और ')     # dictionary
print(bool({t: t}), bool({t: f}), bool({f: f}), sep=' और ')  # dictionary 2.0

print()

print(list(range(t)), list(range(f)), sep=' और ')  # range-list coercion

print()

print(bin(t), bin(f), sep=' और ')  # binary
print(oct(t), oct(f), sep=' और ')  # octal
print(hex(t), hex(f), sep=' और ')  # hexadecimal

print()

from math import inf, nan           # importing infinity and NaN (Not a Number)
print(bool(inf), bool(nan), sep=' और ')
