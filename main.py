import chess
import chess.svg
from flask import Flask, Markup, render_template
import tensorflow as tf
import numpy as np
import math

board = chess.Board()
board.push_san("e4")

print(board)
# app = Flask(__name__)
#
# @app.route('/')
# def test():
#     print(chess.svg.board(board))
#     return render_template('index.html', svg=Markup(chess.svg.board(board, size = 500)))
#
# app.run()
#

input_length = 92
fen_lengths = []
with open("data/chessData.csv") as file:
    file.readline()
    for i in range(0, 100000):
        position_eval = file.readline().split(",")
        # position_eval[1] = float(position_eval[1].replace("\n", ""))
        print(position_eval)
        print(chess.Board(position_eval[0]))
        print("CentiPawn Score:", position_eval[1])
        normalized_evaluation = 1 if "#" in position_eval[1] else 2 * (1 / (1 + math.exp(-float(position_eval[1])/300)) - 0.5)
        print()
        print("Normalized Evaluation:", normalized_evaluation)
        position_eval[0] = position_eval[0] + (input_length - len(position_eval[0])) * " " # Add padding on FEN
print("max Centipawn score:", max(max_centipawn_score))
# tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, input_shape=input_length)])
