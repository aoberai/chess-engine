import chess
import chess.svg
from flask import Flask, Markup, render_template
import tensorflow as tf
import numpy as np
import math

piece_char_2_int = {"R" : 21, "N": 22, "B": 23, "Q" : 24, "K" : 25, "P" : 26, "r" : 11, "n": 12, "b": 13, "q" : 14, "k" : 15, "p" : 16, "." : 0} # Uppercase is Black, lowercase white. Conversion to int since model can only take float input, not string


input_length = 92
fen_lengths = []
positions = []
evaluations = []
colors = []
with open("data/chessData.csv") as file:
    file.readline()
    line_number = 1
    for line in file.readlines():
        position_eval = line.split(",")
        #numpy position_eval[1] = float(position_eval[1].replace("\n", ""))
        normalized_evaluation = 1 if "#" in position_eval[1] else 2 * (1 / (1 + math.exp(-float(position_eval[1])/300)) - 0.5) # normalizes centipawn score with sigmoid function
        position_eval[0] = position_eval[0] + (input_length - len(position_eval[0])) * " " # Add padding on FEN
        position_array = np.asarray([i.split(" ") for i in str(chess.Board(position_eval[0])).split("\n")])
        position_array_int = np.zeros((8, 8))
        for i in range(0, len(position_array)):
            for j in range(0, len(position_array[i])):
                position_array_int[i][j] = piece_char_2_int[position_array[i][j]]
        color = 1 if 'w' == position_eval[0].split()[1] else 0 # 1 if white, 0 if black to move

        print(line_number)
        print(position_eval)
        print(chess.Board(position_eval[0]).unicode())
        print(chess.Board(position_eval[0]))
        print(position_array_int)
        print("CentiPawn Score:", position_eval[1])
        print("Normalized Evaluation:", normalized_evaluation)
        print("Color:", color)

        positions.append(position_array_int)
        evaluations.append(normalized_evaluation)
        colors.append(color)

        line_number += 1
        if line_number == 12000000:
            break
np.save("evaluations", np.asarray(evaluations))
np.save("positions", np.asarray(positions))
np.save("colors", np.asarray(colors))

# Attempt at displaying board svg on web
# board = chess.Board()
# board.push_san("e4")
#
# print(board)
#
# app = Flask(__name__)
#
# @app.route('/')
# def test():
#     print(chess.svg.board(board))
#     return render_template('index.html', svg=Markup(chess.svg.board(board, size = 500)))
#
# app.run()


