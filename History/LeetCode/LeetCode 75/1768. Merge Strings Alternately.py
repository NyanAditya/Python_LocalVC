def mergeAlternately(word1: str, word2: str) -> str:
    out = ''
    i = 0

    while i < min(len(word1), len(word2)):
        out += word1[i] + word2[i]
        i += 1

    if len(word1) > len(word2):
        out += word1[i:]

    else:
        out += word2[i:]

    return out


print(mergeAlternately('ABCD', 'pq'))
