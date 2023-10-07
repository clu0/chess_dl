#!/usr/bin/env python3
import sys
import os
import chess
import chess.pgn
from state import *
import numpy as np

def create_dataset(
    path: str,
    chunksize: int = 10000,
    data_dir: str = "data",
    max_games: int = 10000000) -> None:
    """
    return states, values, actions
    """
    print(f"create_dataset, path {path}, chunk {chunksize}, data_dir {data_dir}, {max_games}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    pgn = open(path)
    states, values, actions = [], [], []
    result_value = {'1/2-1/2': 0., '0-1': -1., '1-0': 1.}
    n_games = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game:
            n_games += 1
            cboard: chess.Board = game.board()
            move_list = list(game.mainline_moves())
            # for first try we're just going to have the current board as the input state
            # n_past_boards = 7
            # past_states = np.zeros((n_past_boards * M, N, N))
            result = game.headers["Result"]
            if result in result_value:
                game_value = result_value[game.headers["Result"]]
            else:
                print(f"unknown result {result}, skipping game {n_games}")
                #continue
            for mv in move_list:
                cur_state = board2numpy(cboard)
                # not including history
                # cur_state = np.concatenate([past_states, board], axis=0)
                next_move = legal_moves_to_mask([mv], cboard.turn)
                states.append(cur_state)
                values.append(game_value)
                actions.append(next_move)
                cboard.push(mv)
                # not including history
                # past_states = cur_state[M :((n_past_boards + 1) * M), :, :]
            if n_games % chunksize == 0:
                print(f"parsed {n_games} games")
                save_path = os.path.join(data_dir, f"game_{n_games - chunksize}_{n_games - 1}.npz")
                np.savez(
                    save_path,
                    states=np.array(states),
                    values=np.array(values),
                    actions=np.array(actions)
                )
                states, values, actions = [], [], []
                if n_games >= max_games:
                    break
        else:
            break

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("usage: python extract_data.py <pgn_file> <chunk_size> <save_dir> <max_games>")
        sys.exit(1)
    create_dataset(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))