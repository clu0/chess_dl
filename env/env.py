import numpy as np
import torch
import chess
import sys


# the goal will be to follow the paper Mastering Chess and Shogi by self-play with a general reinforcement learning algorithm

# this file contains the code that builds the current state of the board, and handles the moves

debug = True

def debug_print(str):
    if debug:
        print(str)


# params that govern the current and past states, which will be a N * N * (M * T + L) tensor
# N is size of the board
# M consists of 6 + 6 features for each players' pieces, and 1 + 1 for repetition counts of the current positions
# L consists of color (1), total move count (1), P1 castling (2), P2 castling (2), and no-progress count (1)
N = 8
M = 14
L = 7
T = 8
nMoves = 73 # 56 Q moves, 8 K moves, 9 P underpromotions, 3 options for 3 directions

class ChessGame():
    def __init__(self):
        self.history = torch.zeros([1, N, N, M+L])
        self.currentBoard = torch.zeros([N, N, M*T + L])

    def setupGame(self, board=None):
        if not board:
            board = setupBoard()

        self.history[0, ...] = board
        self.currentBoard[:, :, (M*(T-1)):] = board
        print("setting up new game")
        print("starting position")
        print(self.history[-1,...])

def setupBoard():
    # the board is 8x8, the order of the dims will be
    # 0-5: P1 pieces: P N B R Q K
    # 6-11: P2 pieces
    # 12-13: 2 repetitions
    # 14-20: color, Total moves, P1 castling, P2 castling, no-progress
    # the board will be oriented to the perspective of the current player
    board = torch.zeros([N, N, M+L])

    # setup pieces
    # we will denote say e6 with index (2,4), so the board matrix actually looks like the real chess board
    #P1
    board[6,:,0]=1 #P
    board[7,1,1]=1 #N
    board[7,6,1]=1 #N
    board[7,2,2]=1 #B
    board[7,5,2]=1 #B
    board[7,0,3]=1 #R
    board[7,7,3]=1 #R
    board[7,3,4]=1 #B
    board[7,4,5]=1 #B
    #P2
    board[1,:,0+6]=1
    board[0,1,1+6]=1
    board[0,6,1+6]=1
    board[0,2,2+6]=1
    board[0,5,2+6]=1
    board[0,0,3+6]=1
    board[0,7,3+6]=1
    board[0,3,4+6]=1
    board[0,4,5+6]=1
    # everything else is zero at the start
    return board

def boardStr(board):
    board = board.numpy()
    curBoard = np.empty((N,N), dtype=str)
    curBoard[:] = '.'
    pieceCount = np.zeros((N,N))
    p = 'PNBRQK'
    for i in range(6):
        for j in range(2):
            bSlice = board[...,j*6+i]
            #if i > 0:
            #    sys.exit(
            if j == 0:
                curBoard[bSlice == 1] = p[i]
            else:
                curBoard[bSlice == 1] = p[i].lower()
            pieceCount[bSlice == 1] += 1
    if np.sum(pieceCount > 1) > 0:
        debug_print(f"the piece counts are {pieceCount}, and there is more than one piece in a location!")
    #curBoard = np.flip(curBoard, 0)
    #print("state of the board representation:")
    #for i in range(N):
    #    print(' '.join(curBoard[i,:]))
    return '\n'.join([' '.join(curBoard[i,:]) for i in range(N)])

# we will use the python chess library, so we need a function that for a given board, gets the list of legal moves from the chess package, and converts that into a binary mask in the shape of the action tensor for the DL engine, of dimension N*N*nMoves

# takes input: a list of legal moves
# turns those into a mask
def toMask(legalMvs):
    mask = torch.zeros([N, N, nMoves], dtype=int)
    for mv in legalMvs:
        fromSq = sqToGd(mv.from_square)
        toSq = sqToGd(mv.to_square)
        diff = (fromSq[0] - toSq[0], toSq[1] - fromSq[1]) # first coord is reversed, because rows of board are flipped
        #debug_print(f"in toMask, fromSq is {fromSq}, toSq is {toSq}, diff is {diff}")
        if mv.promotion:
            if mv.promotion < 5:
                direction = toSq[1] - fromSq[1] + 1 # 0, 1 or 2
                p = mv.promotion - 2
                mask[fromSq[0], fromSq[1], 64 + 3*direction + p] = 1
            else:
                mask[fromSq[0], fromSq[1], toQmv(diff)] = 1
        else:
            if (abs(diff[0]) == 1 and abs(diff[1]) == 2) or (abs(diff[0]) == 2 and abs(diff[1]) == 1):
                mask[fromSq[0], fromSq[1], 56 + toNmv(diff)] = 1
            else:
                mask[fromSq[0], fromSq[1], toQmv(diff)] = 1
    return mask

def sqToGd(sq):
    return [7-int(sq/8), sq%8]

def gdToSq(gd):
    return (7-gd[0])*8 + gd[1]

def toNmv(diff):
    if diff[0] == 2 and diff[1] == 1:
        return 0
    elif diff[0] == 1 and diff[1] == 2:
        return 1
    elif diff[0] == -1 and diff[1] == 2:
        return 2
    elif diff[0] == -2 and diff[1] == 1:
        return 3
    elif diff[0] == -2 and diff[1] == -1:
        return 4
    elif diff[0] == -1 and diff[1] == -2:
        return 5
    elif diff[0] == 1 and diff[1] == -2:
        return 6
    else:
        return 7
    
    
def toQmv(diff):
    dist = max(abs(diff[0]), abs(diff[1])) - 1 #0 to 6
    diag = min(abs(diff[0]), abs(diff[1])) > 0
    s1 = sign(diff[0])
    s2 = sign(diff[1])
    if s1 == 1 and s2 == 0:
        return dist
    elif s1 == 1 and s2 ==1:
        return 7 + dist
    elif s1 == 0 and s2 == 1:
        return 7*2 + dist
    elif s1 == -1 and s2 == 1:
        return 7*3 + dist
    elif s1 == -1 and s2 == 0:
        return 7*4 + dist
    elif s1 == -1 and s2 == -1:
        return 7*5 + dist
    elif s1 == 0 and s2 == -1:
        return 7*6 + dist
    else:
        return 7*7 + dist
    
    
def sign(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1

# try to convert from a mask back to an actual move
def toMove(ind, piece=None, col=0):
    if ind[2] >= 64:
        direction = int((ind[2]-64)/3) - 1 # -1, 0 or 1
        p = ((ind[2]-64) % 3) + 2
        fromSq = gdToSq(ind[:2])
        toSq = ind[:2]
        if col==0:
            toSq[0] -= 1 # - because the index for board rows is reversed
        else:
            toSq[0] += 1
        toSq[1] += direction
        inBound = (toSq[0] >= 0 and toSq[0] <= 7) and (toSq[1] >= 0 and toSq[1] <= 7)
        if inBound:
            return chess.Move(fromSq, gdToSq(toSq), promotion=p), p
        else:
            sys.exit(f'the move with fromSq {fromSq}, toSq {toSq} is out of bounds')
    else:
        fromSq = gdToSq(ind[:2])
        #print(f"not an underpromotion, from sq is {ind[:2]}")
        if ind[2] >= 56:
            diff = fromNmv(ind[2]-56)
        else:
            diff = fromQmv(ind[2])
            #print(f"the move is {ind[2]}, hence a Qmv, and the actual difference in position is {diff}")
        toSq = ind[:2]
        toSq[0] -= diff[0] # - because the index for board rows is reversed
        toSq[1] += diff[1]
        #print(f"the end position is {toSq}")
        inBound = (toSq[0] >= 0 and toSq[0] <= 7) and (toSq[1] >= 0 and toSq[1] <= 7)
        if inBound:
            if piece == 1 and (toSq[0] == 0 or toSq[0] == 7):
                return chess.Move(fromSq, gdToSq(toSq), promotion=5), 5
            else:
                return chess.Move(fromSq, gdToSq(toSq)), None
        else:
            sys.exit(f'the move with fromSq {fromSq}, toSq {toSq} is out of bounds')
    
def fromNmv(mv):
    if mv == 0:
        return [2,1]
    elif mv == 1:
        return [1,2]
    elif mv == 2:
        return [-1,2]
    elif mv == 3:
        return [-2,1]
    elif mv == 4:
        return [-2,-1]
    elif mv == 5:
        return [-1,-2]
    elif mv == 6:
        return [1,-2]
    else:
        return [2, -1]

def fromQmv(mv):
    direction = int(mv/7)
    dist = (mv % 7) + 1
    #print(f"in fromQmv, the direction is {direction}, and dist is {dist}")
    if direction == 0:
        return [dist,0]
    elif direction == 1:
        return [dist,dist]
    elif direction == 2:
        return [0,dist]
    elif direction == 3:
        return [-dist,dist]
    elif direction == 4:
        return [-dist,0]
    elif direction == 5:
        return [-dist,-dist]
    elif direction == 6:
        return [0,-dist]
    else:
        return [dist,-dist]
    
