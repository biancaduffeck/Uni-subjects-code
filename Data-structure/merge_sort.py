def merge_sort(array):
    if(len(array)<=1):
        return array
    metade=(len(array))//2
    firstHalf=merge_sort(array[:metade])
    secondHalf=merge_sort(array[metade:])
    smallerFirst=0
    smallerSecond=0
    mergedArray=[]
    while(smallerFirst<len(firstHalf) and smallerSecond<len(secondHalf)):
        if(firstHalf[smallerFirst]<secondHalf[smallerSecond]):
            mergedArray.append(firstHalf[smallerFirst])
            smallerFirst+=1
        else:
            mergedArray.append(secondHalf[smallerSecond])
            smallerSecond+=1
    if smallerFirst < len(firstHalf):
        mergedArray.extend(firstHalf[smallerFirst:])
    elif smallerSecond < len(secondHalf):
        mergedArray.extend(secondHalf[smallerSecond:])
    return mergedArray
