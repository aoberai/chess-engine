import random, threading, webbrowser
import numpy as np
import copy
import chess
import tensorflow as tf
import time
import chess.svg
from flask import Flask, Markup, render_template, request
import base64

board = chess.Board()

piece_char_2_int = {
       'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
       'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
       'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
       'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
       'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
       'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
       'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
       'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
       'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
       'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
       'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
       'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
       '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
}

app = Flask(__name__)
@app.route("/")
def update_site():
    board_svg = base64.b64encode(chess.svg.board(board).encode('utf-8')).decode('utf-8')
    ret = '<html><head>'
    ret += '<style>input { font-size: 30px; } button { font-size: 30px; }</style>'
    ret += '</head><body>'
    ret += '<img width=750 height=750 src="data:image/svg+xml;base64,%s"></img><br/>' % board_svg
    ret += '<form action="/move"><input name="Move" type="text"></input><input type="submit"   value="Move"></form><br/>'
    # TODO: Put in evaluation

    ret += '<br> <big><big><big>Position Evaluation for White: %0.4f</big></big></big>' % model.predict(serialize_position(board))
    if board.is_checkmate():
        ret += '<br> <big><big><big><big><big>CheckMate!</big></big></big></big></big>'
    return ret

@app.route("/move")
def update_board():
    input_move = request.args.get("Move")
    legal_moves = [board.san(move).lower() for move in list(board.legal_moves)]
    print("Legal Moves: " + str(legal_moves))
    if legal_moves.count(input_move.lower()) == 0:
        return update_site()
    board.push_san(input_move)
    # time.sleep(1)
    computer_move()
    return update_site()

model = tf.keras.models.load_model("chess_engine_LowestLR.h5")

def computer_move():
    if not board.is_checkmate():
        max_evaluation_score = -1
        max_evaluation_score_move = ""
        legal_moves = [board.san(move) for move in list(board.legal_moves)]
        move_eval_scores = {}
        prev_board_copy = None
        for alg_move in legal_moves:
            board_copy = chess.Board(board.fen())
            board_copy.push_san(alg_move)
            one_hot_board = serialize_position(board_copy)
            evaluation_score = model.predict(one_hot_board)[0][0]
            move_eval_scores[alg_move] = evaluation_score
            if evaluation_score >= max_evaluation_score:
                max_evaluation_score_move = alg_move
                max_evaluation_score = evaluation_score

        time.sleep(0.5)

        board.push_san(max_evaluation_score_move)
        # print("Eval scores list: " + str(move_eval_scores)) 
        # print("\nComputer making move: %s with evaluation score %0.3f\n" % (max_evaluation_score_move, max_evaluation_score))
    else:
        print("\n\n\n Checkmate! \n\n\n")

def serialize_position(board):
    letter_position = np.asarray([i.split(" ") for i in str(board).split("\n")]) # gives position using [r, R, k, K, b, B, q, Q, k, K, p, P, .] notation
    one_hot_board = np.zeros((8, 8, 12))
    for i in range(0, len(letter_position)):
        for j in range(0, len(letter_position[i])):
            one_hot_board[i][j] = piece_char_2_int[letter_position[i][j]]
    return np.expand_dims(one_hot_board,0)

if __name__ == "__main__":
    computer_move()
    url = "http://127.0.0.1:5000"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run()


