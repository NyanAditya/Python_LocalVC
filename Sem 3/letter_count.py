def charfreq(x):
    freq = dict()
    
    unique = set([ch for ch in x])
    
    unique = list(unique)
    
    for ch in unique:
        freq[ch] = x.count(ch)
        
    return freq

print(charfreq("PussyCat"))
