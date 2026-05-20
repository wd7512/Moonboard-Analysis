import numpy as np
import matplotlib.pyplot as plt
def convert_key(key):
    '''
    Converts desciption (e.g B10) into a row and column index
    for a 18x11 array
    '''
    letter = key[0]
    number = key[1:]
    
    col = ord(letter) - 65
    row = 18-int(number)

    return row,col

def convert_moves(moves):
    '''
    Converts the list of dictionarys from the moves column into
    3 18x11 arrays for start/middle/end holds
    '''
    holds = np.zeros((3,18,11),dtype=bool)

    
    for dic in moves:
        key = dic['Description']
        
        row,col = convert_key(key)
        if dic['IsStart']:
            holds[0,row,col] = True
        elif dic['IsEnd']:
            holds[2,row,col] = True
        else:
            holds[1,row,col] = True
            
    return holds

def plot_moves(moves):
    
    figure, axis = plt.subplots(1,3)

    axis[0].matshow(moves[0])
    axis[0].set_title('Start Holds')
    axis[1].matshow(moves[1])
    axis[1].set_title('Middle Holds')
    axis[2].matshow(moves[2])
    axis[2].set_title('End Holds')
    
    
def condense(moves):
    '''
    function to remove empty rows of the dataset
    '''
    new_moves = np.zeros((22,11))
    
    new_moves[0:5,:] = moves[0][12:17,:]
    new_moves[5:21,:] = moves[1][1:17,:]
    new_moves[-1,:] = moves[2][0,:]
    
    return new_moves

def uncondense(moves):
    '''
    function to restore condensed data into 3 arrays
    '''
    
    new_moves = np.zeros((3,18,11))
    
    new_moves[0][12:17,:] = moves[0:5,:]
    new_moves[1][1:17,:] = moves[5:21,:]
    new_moves[2][0,:] = moves[-1,:]
    
    return new_moves

def get_null_moves():
    empty_holds = ["F18",
                  "J18",
                  "A17","B17","C17","E17","F17","H17","I17","J17","K17",
                  "J15","K15",
                  "B14",
                  "A8",
                  "A7",
                  "A6","H6",
                  "B5","E5","G5",
                  "A4","C4","D4","E4","F4","H4","J4","K4",
                  "A3","C3","E3","F3","G3","H3","I3","J3","K3",
                  "A2","B2","C2","D2","E2","F2","H2","I2","K2",
                  "A1","B1","C1","D1","E1","F1","G1","H1","I1","J1","K1",]
    null_moves = np.zeros((3,18,11)) #creates empty route to highlight null holds
    
    for hold in empty_holds:
        row,col = convert_key(hold)

        null_moves[0][row,col] = 1
        null_moves[1][row,col] = 1
        null_moves[2][row,col] = 1
        
    null_moves = condense(null_moves)
    null_moves = null_moves.flatten()
        
    return null_moves

def reverse(moves):
    
    moves = list(moves)
    null_moves = get_null_moves()
    
    for i in range(null_moves.size):
        if null_moves[i] == 1:
            moves.insert(i,0)
            
    moves = np.array(moves)
    
    moves = moves.reshape((22,11))
    
    moves = uncondense(moves)
    
    return moves