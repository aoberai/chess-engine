import math
import threading
import webbrowser
import numpy as np
import chess
import tensorflow as tf
import time
import chess.svg
import chess.engine
from flask import Flask, request
import base64

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(
    "/home/aoberai/programming/python/chess-engine/stockfish")
infinity = 1000000000
selfplay = False

app = Flask(__name__)


@app.route("/")
def update_site():
    board_svg = base64.b64encode(chess.svg.board(
        board, flipped=True).encode('utf-8')).decode('utf-8')
    ret = '<html><head>'
    ret += '<style>input { font-size: 30px; } button { font-size: 30px; }</style>'
    ret += '</head><body>'

    ret += '<p style="font-size:30px">'

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("R")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("R")) + "   "

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("N")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("N")) + "   "

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("B")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("B")) + "   "

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("Q")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("Q")) + "   "

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("P")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("P")) + "<br>"

    ret += '<img width=750 height=750 src="data:image/svg+xml;base64,%s"></img><br/>' % board_svg

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("r")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("r")) + "   "

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("n")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("n")) + "   "

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("b")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("b")) + "   "

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("q")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("q")) + "   "

    ret += '<img width=60 height=60 src="data:image/svg+xml;base64,%s"></img>' % base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol("p")).encode('utf-8')).decode('utf-8')

    ret += str(str(board).count("p")) + '</p>'

    if board.is_checkmate():
        ret += '<br> <big><big><big><big><big>CheckMate!</big></big></big></big></big>'
    ret += '<br><br><br><br><form action="/move"><input name="Move" type="text" autofocus="autofocus"></input><input type="submit" value="Move"></form> <br><br><form action="/undo"><input type="submit" name="Undo" value="Undo"><br/>'
    ret += '<br> <big><big><big>Position Evaluation for White: %0.4f</big></big></big>' % model.predict(
        serialize_position(board))

    try:
        ret += '<br> <big><big><big>Stockfish Evaluation for White: %0.4f</big></big></big>' % engine.analyse(
            board, chess.engine.Limit(time=0.4))["score"].white().score()  # normalizes centipawn score with sigmoid function
    except Exception as e:
        print("Stockfish Evaluation Not Working")
    try:
        ret += '<br> <big><big><big>Stockfish Recommended Move: %s </big></big></big>' % engine.play(
            board, chess.engine.Limit(time=0.1)).move
    except Exception as e:
        print("Stockfish Best Move Not Working")
    return ret


@app.route("/move")
def update_board():
    if not selfplay:

        input_move = request.args.get("Move")
        legal_moves = [board.san(move).lower()
                       for move in list(board.legal_moves)]
        print("Legal Moves: " + str([board.san(move)
              for move in list(board.legal_moves)]))
        if legal_moves.count(input_move.lower()) == 0:
            return update_site()
        board.push_san(input_move)

        computer_move(turn=chess.WHITE)
        '''
        Move Recommendations
        legal_moves = [board.san(move) for move in list(board.legal_moves)]
        move_eval_scores = {}
        for alg_move in legal_moves:
            board_copy = chess.Board(board.fen())
            board_copy.push_san(alg_move)
            one_ply_evaluation_score = model.predict(
                serialize_position(board_copy))
            move_eval_scores[alg_move] = one_ply_evaluation_score

        sorted_move_eval_scores = sorted(
            move_eval_scores.items(), key=lambda x: x[1])
        print("1-PLY Move Recommendations")
        print(sorted_move_eval_scores)
        '''
    else:
        legal_moves = [board.san(move).lower()
                       for move in list(board.legal_moves)]
        print("Legal Moves: " + str([board.san(move)
              for move in list(board.legal_moves)]))
        input_move = request.args.get("Move")
        if input_move == '':
            computer_move(turn=board.turn)
        else:
            if legal_moves.count(input_move.lower()) == 0:
                return update_site()
            board.push_san(input_move)
    return update_site()


@app.route("/undo")
def undo_move():
    try:
        board.pop()
        board.pop()
    except Exception as e:
        print("Cannot undo move")
    return update_site()


# model = tf.keras.models.load_model("chess_engine_v4.h5") # best
model = tf.keras.models.load_model("chess_engine_vlatest.h5")


def computer_move(turn=chess.WHITE):
    if not board.is_checkmate():
        legal_moves = [board.san(move) for move in list(board.legal_moves)]
        move_eval_scores = {}
        for alg_move in legal_moves:
            board_copy = chess.Board(board.fen())
            board_copy.push_san(alg_move)
            evaluation_score = minimax(
                board_copy.fen(), depth=2, alpha=-infinity, beta=infinity, last_move=alg_move)

            move_eval_scores[alg_move] = evaluation_score
        sorted_move_eval_scores = sorted(
            move_eval_scores.items(),
            key=lambda x: x[1],
            reverse=True)
        print(sorted_move_eval_scores)
        time.sleep(0.25)
        if turn == chess.WHITE:
            # If white, make the move to maximize white position score
            board.push_san(sorted_move_eval_scores[0][0])
        else:
            # If black, make the move to minimize white position score
            board.push_san(sorted_move_eval_scores[-1][0])
        print("\nComputer making move: %s with evaluation score %0.3f\n" %
              (sorted_move_eval_scores[0][0], sorted_move_eval_scores[0][1]))
    else:
        print("\n\n\n Checkmate! \n\n\n")


# TODO: Might need to make this color dependent for minimax?
def position_evaluation(fen, color=chess.WHITE):
    board_copy = chess.Board(fen)
    one_hot_board = serialize_position(board_copy)
    evaluation_score = model.predict(one_hot_board)[0][0]
    return evaluation_score


def serialize_position(board):
    piece_char_2_int = {
        'p': [1, 0, 0, 0, 0, 0],
        'P': [-1, 0, 0, 0, 0, 0],
        'n': [0, 1, 0, 0, 0, 0],
        'N': [0, -1, 0, 0, 0, 0],
        'b': [0, 0, 1, 0, 0, 0],
        'B': [0, 0, -1, 0, 0, 0],
        'r': [0, 0, 0, 1, 0, 0],
        'R': [0, 0, 0, -1, 0, 0],
        'q': [0, 0, 0, 0, 1, 0],
        'Q': [0, 0, 0, 0, -1, 0],
        'k': [0, 0, 0, 0, 0, 1],
        'K': [0, 0, 0, 0, 0, -1],
        '.': [0, 0, 0, 0, 0, 0],
    }
    # gives position using [r, R, k, K, b, B, q, Q, k, K, p, P, .] notation
    letter_position = np.asarray([i.split(" ")
                                 for i in str(board).split("\n")])
    one_hot_board = np.zeros((8, 8, 6))
    for i in range(0, len(letter_position)):
        for j in range(0, len(letter_position[i])):
            one_hot_board[i][j] = piece_char_2_int[letter_position[i][j]]
    return np.expand_dims(one_hot_board, 0)


def minimax(fen, depth, last_move, alpha, beta, maximizing_player_color=chess.WHITE):  # depth represents ply
    # Establish Search Tree
    board = chess.Board(fen)
    if depth == 0 or board.is_game_over(claim_draw=True):
        # play against custom trained model
        position_evaluation_score = float(position_evaluation(fen))
        # position_evaluation_score = engine.analyse(board,
        # chess.engine.Limit(time=0.05))["score"].white().score() # play
        # against stockfish
        print(board.unicode())
        print(position_evaluation_score)
        print("Move Number: ", board.fullmove_number)
        print("Last Move: ", last_move)
        return position_evaluation_score

    # used as a threshold to find max eval if search node on
    # maximizing_player_color or min eval for oppositve player if search node
    # not on maximizing_player_color
    threshold_evaluation = -infinity if board.turn == maximizing_player_color else infinity
    # if depth == 2:
    # print(board.peek())
    legal_moves = [board.san(move) for move in list(board.legal_moves)]
    maxmin_evaluation = 0
    for move in legal_moves:
        next_board = chess.Board(fen)
        next_board.push_san(move)
        evaluation = minimax(next_board.fen(), depth - 1, move, alpha, beta)
        if board.turn == maximizing_player_color:
            maxmin_evaluation = max(threshold_evaluation, evaluation)
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        else:
            maxmin_evaluation = min(threshold_evaluation, evaluation)
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
    return maxmin_evaluation


if __name__ == "__main__":
    computer_move()
    url = "http://127.0.0.1:5000"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run()
