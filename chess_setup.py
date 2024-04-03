"""
Main logic for the chess engine.
Contains code for evaluation, engine(algorithm) and perft along with automated playing.
"""

import time
import logging
import random
from typing import Tuple
import chess
import chess.engine
import chess.pgn


logging.basicConfig(
    level=logging.INFO,
)


def evaluate_board(board: chess.Board) -> float:
    """
    Returns the evaluation for a given board position.
    """
    white_pieces_value = sum(
        piece_value(piece)
        for piece in board.piece_map().values()
        if piece.color == chess.WHITE
    )
    black_pieces_value = sum(
        piece_value(piece)
        for piece in board.piece_map().values()
        if piece.color == chess.BLACK
    )
    material_advantage = white_pieces_value - black_pieces_value
    return material_advantage


def piece_value(piece: chess.Piece):
    """
    User defined piece values."""
    # Todo : Replace with dictionary to save time
    if piece.piece_type == chess.PAWN:
        return 1
    elif piece.piece_type == chess.KNIGHT:
        return 3.2
    elif piece.piece_type == chess.BISHOP:
        return 3.4
    elif piece.piece_type == chess.ROOK:
        return 5
    elif piece.piece_type == chess.QUEEN:
        return 9
    elif piece.piece_type == chess.KING:
        return 100
    else:
        return 0


def minimax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    node_count: int,
) -> Tuple[float, int]:
    """
    Performs a minimax search on a given board position.

    Args:
        board : The given board configuration(state).
        depth : Depth to which the search still has to go on, stops at zero.
        alpha : Parameter for minimax.
        beta : Parameter for minimax.
        maximizing_player : True for white, helps in working with a standard eval.
        node_count : The number of nodes visited, used for debugging and perft.

    Returns:
        evaluation : The minimax evaluation for the player in the current position.
        node_count : The number of nodes traversed during the search.
    """
    node_count += 1

    if board.is_checkmate() and maximizing_player:
        # white checkmates black
        return 9999, node_count
    elif board.is_checkmate():
        # black checkmates white
        return -9999, node_count
    elif depth == 0 or board.is_game_over():
        # draw or depth reached
        return evaluate_board(board), node_count

    if maximizing_player:
        # engine as white
        max_eval = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            evaluation, node_count = minimax(
                board, depth - 1, alpha, beta, False, node_count
            )
            board.pop()
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            # prune
            # if beta <= alpha:
            #     break
        return max_eval, node_count

    # engine as black
    min_eval = float("inf")
    for move in board.legal_moves:
        board.push(move)
        evaluation, node_count = minimax(
            board, depth - 1, alpha, beta, True, node_count
        )
        board.pop()
        min_eval = min(min_eval, evaluation)
        beta = min(beta, evaluation)
        # prune
        # if beta <= alpha:
        #     break
    return min_eval, node_count


def choose_move(board: chess.Board, depth: int) -> Tuple[chess.Move, int]:
    """
    Chooses the best move out of all possible legal moves in a position.

    Args:
        board : The given board configuration(state).
        depth : Depth to which the search still has to go on, stops at zero.

    Returns:
        best_move : The best possible (legal) move in the given scenario,
            that does not lead to a draw.
        total_nodes : The number of nodes traversed in the search.
    """
    best_move = None
    max_eval = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    total_nodes = 0

    for move in board.legal_moves:
        board.push(move)
        if board.is_repetition(3):
            # detects a threefold repetition, prevents going back to old positions
            # detection of threefold repetition is done by traversing the total list of moves
            board.pop()
            continue
        evaluation, node_count = minimax(board, depth - 1, alpha, beta, False, 0)
        board.pop()

        total_nodes += node_count
        if evaluation > max_eval:
            max_eval = evaluation
            best_move = move

    return best_move, total_nodes


def get_random_move(board: chess.Board):
    """
    Gets a random move from all legal moves in the given position.
    Built for using as a baseline, plays random moves."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)


def play(white_engine, black_engine, time_control: int):
    """
    Simulates a game of chess between the white and black engines, based on the given time control.

    Args :
        white_engine : The player playing as white in the game.
        black_engine : The player playing as black in the game (both can be humans too).
        time_control : The time limit for making a move. Used to enforce a hard limit on engines.

    Returns :
        board.result : The result of the game played between the two players.
        game_moves : UCI representation of all the moves played in the game."""
    board = chess.Board()
    game = chess.pgn.Game()
    game_moves = ""
    current_move = 1
    if time_control:
        print("not implemented yet")
    while not board.is_game_over():
        searched_nodes = 0
        # ! logic works for white so far, need to generalize for black
        if board.turn == chess.WHITE:
            search_start = time.time()
            move, searched_nodes = white_engine(board, 4)
            search_end = time.time()
            # todo : log this instead of printing
            logging.info("move %s}", current_move)
            current_move += 1
            logging.info(
                "nodes : %s  time: %s",
                str(searched_nodes),
                str(search_end - search_start),
            )
        else:
            search_start = time.time()
            move, searched_nodes = black_engine(board, 2)
            search_end = time.time()
            logging.info(
                "nodes : %s  time: %s",
                str(searched_nodes),
                str(search_end - search_start),
            )

        if move in board.legal_moves:
            board.push(move)
            game.add_variation(move)
            game_moves += " " + str(move)
            logging.info(str(move))
        else:
            print(board)
            raise ValueError(f"Illegal move {move} generated by {board.turn} engine")

    game.headers["Result"] = board.result()
    print("Game over")
    print("Result: ", board.result())

    return board.result(), game_moves


def test_against_previous(current, previous, savefile: str, n_games: int):
    """
    Runs automated games between two versions of the engine for n games,
    saves the pgn to a text file.
    """
    f = open(savefile, "a", encoding="utf-8")
    print("Starting ...")

    for game in range(n_games):
        start_time = time.time()
        result, pgn = play(current, previous, None)
        print(pgn)
        f.write(result + pgn + "\n\n")
        end_time = time.time()
        print(
            f"done with game {game+1} in {end_time - start_time} seconds, result = {result}"
        )
    f.close()


def perft(board, depth):
    """
    Does performance testing on the minimax engine.
    Used to verify if move generation works correctly or not, can also be
    used to verify improvements to the engine.
    """
    if depth == 0:
        return 1

    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()

    return nodes


def perft_test(board, depth):
    """
    I don't want to write docs for this."""
    total_nodes = 0
    for d in range(1, depth + 1):
        nodes = perft(board.copy(), d)
        print(f"Depth {d}: {nodes} nodes")
        total_nodes += nodes

    print(f"Total nodes: {total_nodes}")


def iterative_deepening_minimax(board: chess.Board, max_depth: int, time_limit: int):
    """
    Iterative deepening search for minimax.
    Iterative deepening on minimax allows for less positions to be searched,
    also helps in time limit situations.
    """
    start_time = time.time()
    best_move = None
    for depth in range(1, max_depth + 1):
        alpha = float("-inf")
        beta = float("inf")
        for move in board.legal_moves:
            board.push(move)
            # Todo : add node_count, modify according to improved parameters in minimax
            evaluation, _ = minimax(board, depth, alpha, beta, False, 0)
            board.pop()
            if evaluation > alpha:
                alpha = evaluation
                best_move = move

        if time.time() - start_time >= time_limit:
            break
    return best_move


if __name__ == "__main__":
    test_against_previous(
        choose_move, choose_move, "test_runs/v1.1_4ply_vs_v1.1_2ply.txt", 1
    )
