file_loc = "D:\\SelfRepoClone\\Python_LocalVC\\Sem 3\\Rev Content of a File\\textfile.txt"

with open(file_loc, "w+") as file:
    content = file.read()
    
    print(content)
    
    charlist = [x for x in content]
    
    charlist = charlist[::-1]
    
    newcontent = ",".join(charlist)
    
    print(newcontent)
    
    file.seek(0,0)
    
    file.write(newcontent)
