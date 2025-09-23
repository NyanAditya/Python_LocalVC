def longest_word(wordlst):
    
    lenlist = [len(x) for x in wordlst]
    
    maxlen = max(lenlist)
    
    longest = wordlst[lenlist.index(maxlen)]
    
    return longest

print(longest_word(["apple", "banana", "cherry"]))
