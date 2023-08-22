#!/usr/bin/env python3
import sys
import chess
import chess.pgn
from state import *
import numpy as np

def create_dataset(path):
    """
    return states, values, actions
    """
    pgn = open(path)
    states, values, actions = [], [], []
    result_value = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    n_games = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game:
            n_games += 1
            if n_games % 1000 == 0:
                print(f"parsing game {n_games}")
            cboard = game.board()
            move_list = list(game.mainline_moves())
            n_past_boards = 7
            past_states = np.zeros((N, N, n_past_boards * M), dtype=np.uint8)
            result = game.headers["Result"]
            if result in result_value:
                game_value = result_value[game.headers["Result"]]
            else:
                print(f"unknown result {result}, skipping game {n_games}")
                continue
            for mv in move_list:
                board = board2numpy(cboard)
                cur_state = np.concatenate([past_states, board], axis=2)
                next_move = legal_moves_to_mask([mv], cboard.turn)
                if cboard.turn == chess.WHITE:
                    states.append(cur_state)
                else:
                    states.append(np.flip(cur_state, axis=(0, 1)))
                values.append(game_value)
                actions.append(next_move)
                cboard.push(mv) 
                past_states = cur_state[:, :, M :((n_past_boards + 1) * M)]
        else:
            break
    return np.array(states), np.array(values), np.array(actions)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python extract_data.py <pgn_file> <output_file>")
        sys.exit(1)
    states, values, actions = create_dataset(sys.argv[1])
    np.savez(sys.argv[2], states=states, values=values, actions=actions)