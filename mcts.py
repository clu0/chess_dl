"""
Implementing MCTS for playing games

Look at the original alphago paper for MCTS details: mastering the game of go without human knowledge

A few properties that we need:
- The game state should be organized in trees, so we can easily reuse previous exploration info
- each node of the game should have the following info:
    - state s
    - for each avilable action a:
        - N(s, a), the count of times we reached this position
        - Q(s, a), the estimated value at this position
        - W(s, a), the accumulated values from rollouts
        - P(s, a), the policy function given by the NN
        - We don't actually need to keep Q(s, a), it can be calculated from W and N
        - There is also an upper confidence bound U(s, a), which can also be calculated.
- We should resign if the value function at a node gets below v_resign
    - Should we actually do this? Or should we let things play out?
- In the end, we should get a sequence consisting of (s_i, pi_i, z_i), where
    - s_i is the state
    - pi_i is the policy function calculated using N(s_i, a) from the MCTS
    - z_i is the terminal reward from that self playing game
    - Maybe we just keep tract of this in a separate list from the game tree.
- You know what would be super nice:
    - If we could have a single data structure that stores all of the past states
    - and every time we do MCTS, we start from using this super long tree
    - It doesn't really make sense to replay the game from scratch every time we re-run this

An idea: write the nodes of the graph in such a way that they have a backward function
- When we call backward on a node, we automatically update the nodes of the parent function
"""

from typing import Dict, Union
import torch
from torch import nn
import numpy as np
import numpy.typing as npt
import chess
from state import board2numpy, index_to_move


result2val = {
    "1-0": 1,
    "0-1": -1,
    "1/2-1/2": 0,
}

def get_board_str(board): return str(board)
def board_to_tensor(board: chess.Board) -> torch.Tensor: return torch.from_numpy(board2numpy(board))

class MCTSChess:
    def __init__(self, net: nn.Module, c_exp: float):
        self.Ns: Dict[str, torch.Tensor] = {}
        self.Ws: Dict[str, torch.Tensor] = {}
        self.Ps: Dict[str, torch.Tensor] = {}
        self.Mvs: Dict[str, torch.Tensor] = {}  # caches the valid moves at non-leaf states
        self.Es: Dict[str, Union[None, float]] = {}  # caches whether the game ends at non-leaf state, and if so (not None) the value
        self.net = net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net.to(self.device)
        self.c_exp = c_exp

    def search(self, board: chess.Board) -> float:
        """
        Function that performs MCTS and returns the value of the starting board from one tree search.

        The search function will not change the input board (i.e. it will pop off all the moves that it added)
        
        The function recursively finds a leaf node first, and then it obtains the value at the leaf by querying
        the trained neural network.
        
        The board will tell us whether we have repeated moves
        """
        board_str: str = get_board_str(board)
        # did game end?
        if board_str in self.Es:
            if self.Es[board_str] is not None: return self.Es[board_str]
        else:
            outcome: chess.Outcome = board.outcome
            self.Es[board_str] = outcome if outcome is None else result2val[outcome.result()]
                
        if board_str not in self.Ps:  # leaf node
            state = board_to_tensor(board).to(self.device)
            P, v = self.net(state)
            self.Ps[board_str] = P.to('cpu')
            return v
        
        # non leaf node, need to pick an action based on UCB
        # a_t = argmax Q_q + U_a
        # U_a = c * P_a * \sqrt{\sum_b N_b} / (1 + N_a)
        N = self.Ns[board_str]
        #TODO: double check whether we need to flip P if color is black
        U = self.c_exp * self.Ps[board_str] * torch.sqrt(torch.sum(N)) / (1 + N)
        W = self.Ws[board_str]
        Q = W
        if board.turn == chess.BLACK:
            Q = -Q
        Q[N > 0] /= N[N > 0]
        UCB: npt.NDArray[np.float_] = (Q + U).numpy()
        a_ind = np.unravel_index(np.argmax(UCB), UCB.shape)
        
        # need to get the piece ind and the color for the piece we're moving
        board_np = board2numpy(board)
        piece_inds = board[a_ind[0], a_ind[1], :12]
        pieceInd = np.where(piece_inds == 1)[0].item()
        piece = (pieceInd % 6) + 1
        mv = index_to_move(a_ind, piece == 1, board.turn)

        # make a move on the board, and get the value from the resulting state
        board.push(mv)  #TODO i think this works but need to check
        v = self.search(board)
        W[a_ind] += v
        N[a_ind] += 1
        self.Ws[board_str] = W
        self.Ns[board_str] = N
        board.pop(mv)
        return v
