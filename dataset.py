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
positions = []
evaluations = []
with open("data/chessData.csv") as file:
    file.readline()
    line_number = 1
    for line in file.readlines():
        try:
            position_eval = line.split(",")
            #numpy position_eval[1] = float(position_eval[1].replace("\n", ""))
            normalized_evaluation = 1 if "#" in position_eval[1] else 2 * (1 / (1 + math.exp(-float(position_eval[1])/300)) - 0.5) # normalizes centipawn score with sigmoid function
            position_eval[0] = position_eval[0] + (input_length - len(position_eval[0])) * " " # Add padding on FEN
            print(line_number)
            print(position_eval)
            print(chess.Board(position_eval[0]))
            print("CentiPawn Score:", position_eval[1])
            print("Normalized Evaluation:", normalized_evaluation)
            positions.append(np.asarray([i.split(" ") for i in str(chess.Board(position_eval[0])).split("\n")]))
            evaluations.append(normalized_evaluation)
            line_number += 1
        except Exception as e:
            print(e)
            print("Problem with line:", line_number)
        if line_number == 10000:
            break
# tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, input_shape=input_length)])
np.save("evaluations", np.asarray(evaluations))
np.save("positions", np.asarray(positions))
