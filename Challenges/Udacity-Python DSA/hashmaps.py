"""Write a HashTable class that stores strings
in a hash table, where keys are calculated
using the first two letters of the string."""

class HashTable(object):
    def __init__(self):
        self.table = [None]*10000

    def store(self, string):
        string_hash_code = self.calculate_hash_value(string)
        self.table[string_hash_code] = string
        

    def lookup(self, string):
        string_hash_code = self.calculate_hash_value(string)
        
        if self.table[string_hash_code] != None:
            return string_hash_code

        else:
            return -1
        
        
        
    def calculate_hash_value(self, string):
        hash_code = int(str(ord(string[0])) + str(ord(string[1])))
        return hash_code
    
# Setup
hash_table = HashTable()

# Test calculate_hash_value
# Should be 8568
print hash_table.calculate_hash_value('UDACITY')

# Test lookup edge case
# Should be -1
print hash_table.lookup('UDACITY')

# Test store
hash_table.store('UDACITY')
# Should be 8568
print hash_table.lookup('UDACITY')

# Test store edge case
hash_table.store('UDACIOUS')
# Should be 8568
print hash_table.lookup('UDACIOUS')
