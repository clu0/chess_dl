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

from typing import Dict, Union, Optional, List, Any, Tuple
from dataclasses import dataclass, field
import torch
from torch import nn
import numpy as np
import numpy.typing as npt
import chess
from state import board2numpy, mask_to_lists, legal_moves_to_mask
from stockfish import Stockfish


result2val = {
    "1-0": 1,
    "0-1": -1,
    "1/2-1/2": 0,
}

def get_board_str(board: chess.Board) -> str:
    if board.is_repetition(count=2):
        return board.fen() + " 2"
    elif board.is_repetition():
        return board.fen() + " 3"
    else:
        return board.fen() + " 1"
def board_to_tensor(board: chess.Board) -> torch.Tensor: return torch.from_numpy(board2numpy(board))


@dataclass
class StateInfo:
    Ps: npt.NDArray[np.float_]  # probability mask
    moves: List[chess.Move]  # list of moves
    Ns: npt.NDArray[np.int_]
    Ws: npt.NDArray[np.float_]
    end: Optional[float] = None

class MCTSChess:
    def __init__(self, net: Union[nn.Module, "FakeChessNet"], c_exp: float):
        self.states: Dict[str, "StateInfo"] = {}
        self.c_exp = c_exp
        self.net = net
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(self.net, nn.Module):
            self.net.to(self.device)

    
    def make_move(self, board: chess.Board, temp: float = 1, n_sim: int = 200) -> chess.Move:
        for _ in range(n_sim):
            _ = self.search(board)
        
        board_str: str = get_board_str(board)
        assert board_str in self.states

        state_info: StateInfo = self.states[board_str]
        counts = state_info.Ns
        counts = counts ** (1 / temp)
        probs = counts / counts.sum()
        a_ind =  np.random.choice(len(probs), p=probs)
        return state_info.moves[a_ind]
        

    def search(self, board: chess.Board) -> float:
        """
        Function that performs MCTS and returns the value of the starting board from one tree search.

        The search function will not change the input board (i.e. it will pop off all the moves that it added)
        The function recursively finds a leaf node first, and then it obtains the value at the leaf by querying
        the trained neural network.
        
        The board will tell us whether we have repeated moves
        """
        board_str: str = get_board_str(board)
        if board_str not in self.states:  # leaf node
            outcome = board.outcome()
            end = None if outcome is None else result2val[outcome.result()]
            if end is not None:
                self.states[board_str] = StateInfo(
                    Ps=np.zeros(0),
                    moves=[],
                    Ns=np.zeros(0, dtype=int),
                    Ws=np.zeros(0),
                    end=end,
                )
                return end
            if isinstance(self.net, nn.Module):  # using a real neural net
                state = board_to_tensor(board[np.newaxis, ...]).to(self.device)
                P_tensor, v = self.net(state)
            else: # using the fake network that uses stockfish
                P_tensor, v = self.net(board)
            P_tensor = P_tensor.detach().cpu().numpy()[0, ...]
            v = v.detach().item()
            moves, P = mask_to_lists(P_tensor, board)
            self.states[board_str] = StateInfo(
                Ps=P,
                moves=moves,
                Ns=np.zeros(len(P), dtype=int),
                Ws=np.zeros(len(P)),
                end=end,
            )
            return v

        # did game end?
        state_info = self.states[board_str]
        if state_info.end is not None:
            return state_info.end
        
        # non leaf node, need to pick an action based on UCB
        # a_t = argmax Q_q + U_a
        # U_a = c * P_a * \sqrt{\sum_b N_b} / (1 + N_a)
        Ns = state_info.Ns
        Us = self.c_exp * state_info.Ps * np.sqrt(Ns.sum()) / (1 + Ns)
        Ws = state_info.Ws
        Qs = Ws.copy()
        if board.turn == chess.BLACK:
            Qs = -Qs
        Qs[Ns > 0] /= Ns[Ns > 0]
        UCB: npt.NDArray[np.float_] = Qs + Us
        a_ind = np.argmax(UCB)
        mv = state_info.moves[a_ind]

        # make a move on the board, and get the value from the resulting state
        board.push(mv)  #TODO i think this works but need to check
        v = self.search(board)
        Ws[a_ind] += v
        Ns[a_ind] += 1
        _ = board.pop()
        return v

        
class FakeChessNet:
    """
    Creates a fake chess engine that actually uses Stockfish under the hood to generate probabilities
    """
    def __init__(self, depth: int = 15):
        self.stockfish: Stockfish = Stockfish()
        self.stockfish.set_depth(depth)
        
    def __call__(self, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        self.stockfish.set_fen_position(board.fen())
        evaluation = self.stockfish.get_evaluation()
        if evaluation["type"] == "cp":
            # stockfish eval in centipawn
            # we want to squash to [-1, 1]
            # and rescale a +3 to 1
            v = np.tanh(evaluation["value"] / 300)
        else:
            assert evaluation["type"] == "mate"
            v = 1 if board.turn == chess.WHITE else -1
        top_moves = self.stockfish.get_top_moves(5)
        moves: List[chess.Move] = [chess.Move.from_uci(move["Move"]) for move in top_moves]
        prob_mask: npt.NDArray[np.float_] = legal_moves_to_mask(
            moves,
            board.turn,
        )
        prob_mask /= len(moves)  # normalize into probability distribution
        prob_mask = prob_mask[np.newaxis, ...]
        return torch.Tensor(prob_mask), torch.Tensor([v])
