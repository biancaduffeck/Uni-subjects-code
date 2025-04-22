def tower_of_hanoi(count, stacks=None, source=0, auxiliary=1, destination=2, moves=0):
    if not stacks:
        stacks = [['ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in range(count-1, -1, -1)], [], []]
        moves = 1
        print(stacks)
    
    if(count==1):
        if(stacks[source]):
            stacks[destination].append(stacks[source].pop())
            print(stacks)
            moves+=1
            
    else:
        moves=tower_of_hanoi(count-1,stacks,source,destination,auxiliary,moves)
        stacks[destination].append(stacks[source].pop())
        print(stacks)
        moves+=1
        moves=tower_of_hanoi(count-1,stacks,auxiliary,source,destination,moves)

    return moves


print(tower_of_hanoi(2))