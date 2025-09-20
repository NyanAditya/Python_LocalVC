with open("./textfile.txt", "w+") as file:
    content = file.read()
    
    print(content)
    
    charlist = [x for x in content]
    
    charlist = charlist[::-1]
    
    newcontent = ",".join(charlist)
    
    print(newcontent)
    
    file.seek(0,0)
    
    file.write(newcontent)
