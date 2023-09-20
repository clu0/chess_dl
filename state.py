from typing import Any
import sys

import numpy as np
import numpy.typing as npt
import torch
import chess


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
EPS = 1e-8

class ChessGame():
    def __init__(self):
        self.history = torch.zeros([1, N, N, M+L])
        self.currentBoard = torch.zeros([N, N, M*T + L])

    def setupGame(self, board=None):
        if not board:
            board = setup_board()

        self.history[0, ...] = board
        self.currentBoard[:, :, (M*(T-1)):] = board
        print("setting up new game")
        print("starting position")
        print(self.history[-1,...])

def setup_board():
    # the board is 8x8, the order of the dims will be
    # 0-5: P1 pieces: P N B R Q K
    # 6-11: P2 pieces
    # 12-13: 2 repetitions
    # 14-20: color, Total moves, P1 castling, P2 castling, no-progress
    # the board will be oriented to the perspective of the current player
    board = np.zeros((N, N, M+L), dtype=np.int8)

    # setup pieces
    # we will denote say e6 with index (2,4), so the board matrix actually looks like the real chess board
    #P1
    board[1,:,0]=1 #P
    board[0,1,1]=1 #N
    board[0,6,1]=1 #N
    board[0,2,2]=1 #B
    board[0,5,2]=1 #B
    board[0,0,3]=1 #R
    board[0,7,3]=1 #R
    board[0,3,4]=1 #Q
    board[0,4,5]=1 #K
    #P2
    board[6,:,0+6]=1
    board[7,1,1+6]=1
    board[7,6,1+6]=1
    board[7,2,2+6]=1
    board[7,5,2+6]=1
    board[7,0,3+6]=1
    board[7,7,3+6]=1
    board[7,3,4+6]=1
    board[7,4,5+6]=1
    # everything else is zero at the start
    # except castling rights, which are set to 1 below
    board[..., (M+2):(M+6)] = 1
    return board

def board_str(board):
    """
    takes a board tensor or array and returns a string representation of the board
    """
    if torch.is_tensor(board):
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
    return '\n'.join([' '.join(curBoard[i,:]) for i in reversed(range(N))])


# we will use the python chess library, so we need a function that for a given board,
# gets the list of legal moves from the chess package, 
# and converts that into a binary mask in the shape of the action tensor for the DL engine, of dimension N*N*nMoves

def legal_moves_to_mask(legal_moves, color=chess.WHITE):
    """
    takes input: list of Move objects
    returns a numpy binary mask of shape (N, N, nMoves)
    where 1 indicates a legal move
    
    See alphazero paper for description of nMoves (73) and how it is decoded
    essentially 56 queen moves first, then knight moves, then underpromotions

    We are flipping the boards when color is black
    So we will encode the move in the flipped direction
    """
    mask = np.zeros([N, N, nMoves])
    for mv in legal_moves:
        from_sq = sq2gd(mv.from_square)
        to_sq = sq2gd(mv.to_square)
        if color == chess.BLACK:
            from_sq = 7 - from_sq
            to_sq = 7 - to_sq
        diff = [to_sq[0] - from_sq[0], to_sq[1] - from_sq[1]]
        #debug_print(f"in legal_moves_to_mask, mv is {mv}, from_sq is {from_sq}, to_sq is {to_sq}, diff is {diff}")
        if mv.promotion:
            if mv.promotion < 5:
                direction = to_sq[1] - from_sq[1] + 1 # 0, 1 or 2
                p = mv.promotion - 2
                move_ind = 64 + 3*direction + p
            else:
                move_ind = toQmv(diff)
        else:
            if (abs(diff[0]) == 1 and abs(diff[1]) == 2) or (abs(diff[0]) == 2 and abs(diff[1]) == 1):
                move_ind = 56 + toNmv(diff)
            else:
                move_ind = toQmv(diff)
        mask[from_sq[0], from_sq[1], move_ind] = 1
    return mask.transpose((2, 0, 1))


def sq2gd(sq):
    return np.array([sq // 8, sq % 8], dtype=np.int32)

def gd2sq(gd):
    return gd[0] * 8 + gd[1]

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

def index_to_move(ind, is_pawn=False, color=chess.WHITE):
    """
    Convert a np binary index array, with 3 indices, indexing the (N, N, nMoves) possible moves,
    into a chess.Move object
    
    We will flip the move direction when color is black
    """
    from_sq = ind[:2].copy()
    to_sq = ind[:2].copy()
    if ind[2] >= 64:
        direction = int((ind[2]-64)/3) - 1 # -1, 0 or 1
        promotion = ((ind[2]-64) % 3) + 2
        #if color == chess.WHITE:
        to_sq[0] += 1 # - because the index for board rows is reversed
        #else:
        #    to_sq[0] -= 1
        to_sq[1] += direction
    else:
        #print(f"not an underpromotion, from sq is {ind[:2]}")
        if ind[2] >= 56:
            diff = fromNmv(ind[2]-56)
        else:
            diff = fromQmv(ind[2])
        to_sq[0] += diff[0] # - because the index for board rows is reversed
        to_sq[1] += diff[1]
        if is_pawn and to_sq[0] in [0, 7]:
            promotion = 5
        else:
            promotion = None
    if not check_in_bound(to_sq):
        raise ValueError(f'the move with from_sq {from_sq}, to_sq {to_sq} is out of bounds')
    if color == chess.BLACK:
        from_sq = 7 - from_sq
        to_sq = 7 - to_sq
    return chess.Move(gd2sq(from_sq), gd2sq(to_sq), promotion=promotion)

    
def check_in_bound(sq):
    return (sq[0] >= 0 - EPS and sq[0] <= 7 + EPS) and (sq[1] >= 0 - EPS and sq[1] <= 7 + EPS)
    

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
    

def board2numpy(cboard):
    """_summary_
    convert a chess board object to a numpy array of shape (N,N,M+L)
    See alphazero paper for description of N, M, L
    """
    assert cboard.is_valid()
    state: npt.NDArray[Any] = np.zeros((N, N, M+L))
    # pieces
    piece2int = {p: i for i, p in enumerate("PNBRQKpnbrqk")}
    for i in range(N * N):
        piece = cboard.piece_at(i)
        if piece is not None:
            state[i // N, i % N, piece2int[piece.symbol()]] = 1
    color = 1 - int(cboard.turn)  # 0 for while, 1 for black
    # repetitions
    state[..., M-2+color] = cboard.is_repetition(2)
    # color
    state[..., M] = color
    # total moves
    state[..., M + 1] = cboard.fullmove_number
    # P1 castling
    state[..., M + 2] = cboard.has_kingside_castling_rights(chess.WHITE)
    state[..., M + 3] = cboard.has_queenside_castling_rights(chess.WHITE)
    # P2 castling
    state[..., M + 4] = cboard.has_kingside_castling_rights(chess.BLACK)
    state[..., M + 5] = cboard.has_queenside_castling_rights(chess.BLACK)
    # no-progress count, recorded in half-moves (100 half-moves means 50 moves rule kicks in)
    state[..., M + 6] = cboard.halfmove_clock
    return state.transpose((2, 0, 1))