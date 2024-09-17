# A program to implement Quick sort

def quicksort(array): # Work in PLace
    
    def recursiveFixer(low, high): # Fixes all the other elements left and right of the Pivot
        if low < high: # Excludes the cases of single element partitions
            pivotIndex = elementFixer(low, high)
            
            recursiveFixer(low, pivotIndex-1)
            recursiveFixer(pivotIndex+1, high)
        
    
    def elementFixer(low, high): # Fixes the pos of the Pivot wrt Sorted Array
        pivot = array[high]
        i = low-1
        
        for j in range(low, high):
            if array[j] < pivot:
                i += 1
                array[i], array[j] = array[j], array[i]
                
        array[i+1], array[high] = array[high], array[i+1]
        
        return i+1
    
    recursiveFixer(0, len(array)-1)


test = [21, 4, 1, 3, 9, 20, 25, 6, 21, 14]
quicksort(test)
print(test)
