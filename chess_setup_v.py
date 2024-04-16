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
import chess.polyglot

mvv_lva = [
    [0, 0, 0, 0, 0, 0, 0],  # victim None
    [15, 14, 13, 12, 11, 10, 0],  # victim P, attacker P, N, B, R, Q, K, None
    [25, 24, 23, 22, 21, 20, 0],  # victim N, attacker P, N, B, R, Q, K, None
    [35, 34, 33, 32, 31, 30, 0],  # victim B, attacker P, N, B, R, Q, K, None
    [45, 44, 43, 42, 41, 40, 0],  # victim R, attacker P, N, B, R, Q, K, None
    [55, 54, 53, 52, 51, 50, 0],  # victim Q, attacker P, N, B, R, Q, K, None
    [0, 0, 0, 0, 0, 0, 0],  # victim K, attacker P, N, B, R, Q, K, None
]

logging.basicConfig(
    level=logging.INFO,
)

SAVEFILE = "test_runs/viraj.txt"


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

    # ! mobility might be a bad idea for evaluation

    mobility = len(list(board.legal_moves)) / 10
    # Todo : change all this stuff into a class instead, make the list of legal moves a class var
    # asymmetric eval to be used
    return material_advantage + mobility


def piece_value(piece: chess.Piece):
    """
    User defined piece values."""
    # Todo : Replace with dictionary to save time
    if piece.piece_type == chess.PAWN:
        return 1
    if piece.piece_type == chess.KNIGHT:
        return 3.2
    if piece.piece_type == chess.BISHOP:
        return 3.4
    if piece.piece_type == chess.ROOK:
        return 5
    if piece.piece_type == chess.QUEEN:
        return 9
    if piece.piece_type == chess.KING:
        return 100
    else:
        return 0


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


def order_moves(board: chess.Board, moves: List[chess.Move]):
    def comp(board: chess.BaseBoard, move: chess.Move):
        vic = board.piece_at(move.to_square)
        atk = board.piece_at(move.from_square)

        if vic == None or atk == None:
            return 0
        return mvv_lva[vic.piece_type][atk.piece_type]

    moves.sort(reverse=True, key=lambda move: comp(board, move))


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
    if board.is_checkmate():
        # black checkmates white
        return -9999, node_count
    if depth == 0 and maximizing_player is True and not board.is_check():
        q_eval, q_count = quiescence_search(board, alpha, beta, 0)
        if q_eval > evaluate_board(board):
            return q_eval, node_count + q_count
    if depth == 0 or board.is_game_over():
        # draw or depth reached
        return evaluate_board(board), node_count
    if maximizing_player:
        # playing as white
        max_eval = float("-inf")
        moves = list(board.legal_moves)
        order_moves(board, moves)
        for move in moves:
            board.push(move)
            evaluation, node_count = minimax(
                board, depth - 1, alpha, beta, False, node_count
            )
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
        evaluation, node_count = minimax(
            board, depth - 1, alpha, beta, True, node_count
        )
        board.pop()
        min_eval = min(min_eval, evaluation)
        beta = min(beta, evaluation)
        # prune
        if beta <= alpha:
            break
    return min_eval, node_count


def quiescence_search(
    board: chess.Board, alpha: float, beta: float, node_count: int
) -> Tuple[float, int]:

    node_count += 1

    stand_pat = evaluate_board(board)
    if stand_pat >= beta:
        return beta, node_count
    alpha = max(alpha, stand_pat)
    moves = list(board.legal_moves)
    order_moves(board, moves)
    for move in moves:
        if board.is_capture(move) or board.gives_check(move):
            board.push(move)
            score, node_count = quiescence_search(board, -beta, -alpha, node_count)
            score = -score
            board.pop()

            if score >= beta:
                return beta, node_count
            alpha = max(alpha, score)

    return alpha, node_count


def choose_move(
    board: chess.Board, depth: int, positions: set
) -> Tuple[chess.Move, int]:
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
    total_nodes = 0
    max_eval = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    moves = list(board.legal_moves)
    best_move = moves[0]
    order_moves(board, moves)
    for move in moves:
        board.push(move)
        if chess.polyglot.zobrist_hash(board) in positions:
            board.pop()
            continue
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

    board.push(best_move)
    positions.add(chess.polyglot.zobrist_hash(board))
    board.pop()
    return best_move, total_nodes


def get_random_move(board: chess.Board):
    """
    Gets a random move from all legal moves in the given position.
    Built for using as a baseline, plays random moves."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)


def play_opening(move_number: int, maximizing_player: bool):
    """
    Plays the Vienna gambit opening to prevent pieces from shuffling around.
    Acts as a temporary opening book."""
    if maximizing_player:
        if move_number == 1:
            return "e2e4"
        if move_number == 2:
            return "b1c3"
        if move_number == 3:
            return "f2f4"
    if move_number == 1:
        return "e7e5"
    if move_number == 2:
        return "g8f6"
    if move_number == 3:
        return "e5f4"
    raise ValueError("Incorrect move number for opening")


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
    positions = set()
    game_moves = ""
    current_move = 1
    if time_control:
        print("not implemented yet")
    while not board.is_game_over():
        searched_nodes = 0
        # ! work out why the engine does not go on the attack
        if board.turn == chess.WHITE:
            search_start = time.time()
            move, searched_nodes = white_engine(board, 2, positions)
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
            move, searched_nodes = black_engine(board, 3, positions)
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
    test_against_previous(choose_move, choose_move, 1)
