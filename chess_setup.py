"""
Main logic for the chess engine.
Contains code for evaluation, engine(algorithm) and perft along with automated playing.
"""

import time
import logging
import random
from typing import Tuple, List
import chess
import chess.engine
import chess.pgn


logging.basicConfig(
    level=logging.INFO,
)

SAVEFILE = "test_runs/iterative_vs_baseline.txt"
piece_values = {
    chess.PAWN: 1,
    chess.ROOK: 5,
    chess.BISHOP: 3.4,
    chess.KNIGHT: 3.2,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def evaluate_board(board: chess.Board) -> float:
    """
    Returns the evaluation for a given board position.
    """
    pieces = list(board.piece_map().values())
    white_pieces_value = sum(
        piece_values.get(piece, 0) for piece in pieces if piece.color == chess.WHITE
    )
    black_pieces_value = sum(
        piece_values.get(piece, 0) for piece in pieces if piece.color == chess.BLACK
    )
    material_advantage = white_pieces_value - black_pieces_value

    # ! mobility might be a bad idea for evaluation

    mobility = len(list(board.legal_moves)) / 10
    # Todo : change all this stuff into a class instead, make the list of legal moves a class var
    if not board.turn:
        # black playing, black's material advantage + mobility difference
        return -(material_advantage + mobility)
    return material_advantage + mobility


def get_move_priority(move: chess.Move, board: chess.Board):
    """
    Gets move priority order.
    Move priority order - checkmates first, then checks, followed by captures, then other moves.
    This priority is to be used to sort the moves to get a good move ordering, improving the
    speed of search(better pruning).
    """
    if board.is_checkmate(move):
        return 0
    if board.is_check(move):
        return 1
    if board.is_capture(move):
        return 2
    return 3


def count_sort_moves(moves: List[chess.Move], board: chess.Board) -> List[chess.Move]:
    """
    Orders moves to improve pruning in ab search."""
    sorted_moves = []
    for priority in range(4):
        for move in moves:
            if get_move_priority(move, board) == priority:
                sorted_moves.append(move)
    return sorted_moves


def minimax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    node_count: int,
) -> Tuple[float, int]:
    """
    Performs a minimax search on a given board position.

    Args:
        board : The given board configuration(state).
        depth : Depth to which the search still has to go on, stops at zero.
        alpha : Parameter for minimax.
        beta : Parameter for minimax.
        node_count : The number of nodes visited, used for debugging and perft.

    Returns:
        evaluation : The minimax evaluation for the player in the current position.
        node_count : The number of nodes traversed during the search.
    """
    node_count += 1

    if board.is_checkmate() and board.turn:
        # white checkmates black
        return 9999, node_count
    if board.is_checkmate():
        # black checkmates white
        return -9999, node_count
    if depth == 0 or board.is_game_over():
        # draw or depth reached
        return evaluate_board(board), node_count

    if board.turn:
        # playing as white
        max_eval = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            evaluation, node_count = minimax(board, depth - 1, alpha, beta, node_count)
            board.pop()
            max_eval = max(max_eval, evaluation)
            alpha = max(alpha, evaluation)
            # prune
            if beta <= alpha:
                break
        return max_eval, node_count

    # playing as black
    min_eval = float("inf")
    for move in board.legal_moves:
        board.push(move)
        evaluation, node_count = minimax(board, depth - 1, alpha, beta, node_count)
        board.pop()
        min_eval = min(min_eval, evaluation)
        beta = min(beta, evaluation)
        # prune
        if beta <= alpha:
            break
    return min_eval, node_count


def choose_move_minimax(board: chess.Board, depth: int) -> Tuple[chess.Move, int]:
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
        if board.is_repetition(2):
            # Detects a threefold repetition, prevents going back to old positions.
            # Detection of threefold repetition is done by traversing the total list of moves.
            # If arg to is_repetition is 2, it prevents any position from repeating ever again
            # this might cause some issues on its own as well.
            board.pop()
            continue
        evaluation, node_count = minimax(board, depth - 1, alpha, beta, 0)
        board.pop()

        total_nodes += node_count
        if evaluation > max_eval:
            max_eval = evaluation
            best_move = move

    return best_move, total_nodes


def choose_move_iterative_minimax(
    board: chess.Board, depth: int
) -> Tuple[chess.Move, int]:
    """
    Same as the previous function, this one uses iterative minimax instead.
    """
    best_move = None
    max_eval = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    total_nodes = 0

    for move in board.legal_moves:
        board.push(move)
        if board.is_repetition(2):
            board.pop()
            continue
        evaluation, node_count = minimax(board, depth - 1, alpha, beta, 0)
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


openings = {"Vienna Gambit": ["e2e4", "e7e5", "b1c3", "g8f6", "f2f4", "e5f4"]}


def play_opening(move_number: int, maximizing_player: bool, opening: str):
    """
    Plays an opening to prevent pieces from shuffling around.
    Acts as a temporary opening book."""
    if maximizing_player:
        return openings[opening][2 * (move_number - 1)]
    return openings[opening][2 * (move_number - 1) + 1]


def save_pgn(result: str, pgn: str, savefile: str):
    """
    Saves the result and pgn of a game to a file.

    Args:
        result: The result of the game.
        pgn: The pgn of the game.
        savefile: The file to save the result and pgn to.
    """
    with open(savefile, "a", encoding="utf-8") as f:
        f.write(result + pgn + "\n\n")


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
        print("not implemented time control yet")
    while not board.is_game_over():
        searched_nodes = 0
        # ! work out why the engine does not go on the attack
        if board.turn == chess.WHITE:
            search_start = time.time()
            move, searched_nodes = white_engine(board, 3, 10)
            search_end = time.time()

            logging.info("move %s", current_move)
            current_move += 1
            logging.info(
                "nodes : %s  time: %s",
                str(searched_nodes),
                str(search_end - search_start),
            )
        else:
            search_start = time.time()
            move, searched_nodes = black_engine(board, 3)
            search_end = time.time()
            logging.info(
                "nodes : %s  time: %s\n",
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
            save_pgn(
                result="Stopped due to Invalid move", pgn=game_moves, savefile=SAVEFILE
            )
            raise ValueError(f"Illegal move {move} generated by {board.turn} engine")

    game.headers["Result"] = board.result()
    print("Game over")
    print("Result: ", board.result())

    return board.result(), game_moves


def test_against_previous(current, previous, n_games: int):
    """
    Runs automated games between two versions of the engine for n games,
    saves the pgn to a text file.
    """
    print("Starting ...")

    for game in range(n_games):
        start_time = time.time()
        result, pgn = play(current, previous, None)
        save_pgn(result, pgn, savefile=SAVEFILE)
        end_time = time.time()
        print(
            f"done with game {game+1} in {end_time - start_time} seconds, result = {result}"
        )


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
    nodes_searched = 0
    for depth in range(1, max_depth + 1):
        alpha = float("-inf")
        beta = float("inf")
        for move in board.legal_moves:
            board.push(move)
            # Todo : add node_count, modify according to improved parameters in minimax
            evaluation, nodes_added = minimax(board, depth, alpha, beta, 0)
            board.pop()
            nodes_searched += nodes_added + 1
            if evaluation > alpha:
                alpha = evaluation
                best_move = move

        if time.time() - start_time >= time_limit:
            break
    return best_move, nodes_searched


def play_puzzles(puzzle_filepath: str, plies_to_play: int, search_depth: int):
    output_file_white = open("moves_3_ply_white.txt", "w")
    output_file_black = open("moves_3_ply_black.txt", "w")

    puzzle_file = open(puzzle_filepath)
    puzzle_lines = puzzle_file.readlines()
    puzzle_file.close()
    for puzzle_count, fen_str in enumerate(puzzle_lines):
        logging.info("\npuzzle number %s", puzzle_count + 1)


        # no idea if you should use copies here instead of creating boards
        white_board = chess.Board(fen_str)
        black_board = chess.Board(fen_str)

        moves_played_white = ""
        moves_played_black = ""

        print('playing as white')
        # play as white
        white_board.turn = chess.WHITE
        for _ in range(plies_to_play):
            move, nodes_searched = choose_move_minimax(white_board, search_depth)
            logging.info("%s plays %s %s", white_board.turn, move, nodes_searched)
            if move:
                white_board.push(move)
                moves_played_white += " " + str(move)
            else:
                print("checkmate status:", white_board.is_checkmate())
                print("stalemate status:", white_board.is_stalemate())
                print()
                break
        output_file_white.write(moves_played_white + "\n")

        print('playing as black')
        # play as black
        black_board.turn = chess.BLACK
        for _ in range(plies_to_play):
            move, nodes_searched = choose_move_minimax(black_board, search_depth)
            logging.info("%s plays %s %s", black_board.turn, move, nodes_searched)
            if move:
                black_board.push(move)
                moves_played_black += " " + str(move)
            else:
                print("checkmate status:", black_board.is_checkmate())
                print("stalemate status:", black_board.is_stalemate())
                print()
                break
        output_file_black.write(moves_played_black + "\n")

    output_file_white.close()
    output_file_black.close()


if __name__ == "__main__":
    # test_against_previous(iterative_deepening_minimax, choose_move_minimax, 1)
    play_puzzles("puzzles.txt", 8, 3)
