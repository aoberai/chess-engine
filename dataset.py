import random
import chess
import chess.svg
import numpy as np
import math

# piece_char_2_int = {"R" : 21, "N": 22, "B": 23, "Q" : 24, "K" : 25, "P" : 26, "r" : 11, "n": 12, "b": 13, "q" : 14, "k" : 15, "p" : 16, "." : 0} # Uppercase is Black, lowercase white. Conversion to int since model can only take float input, not string
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


input_length = 92
fen_lengths = []
wanted_dataset_size = 3000000
positions = np.zeros((wanted_dataset_size, 8, 8, 12))
evaluations = np.zeros(wanted_dataset_size)
# colors = []
with open("data/chessData.csv") as file1, open("data/random_evals.csv") as file2:
    file1.readline(); file2.readline()
    combined_raw_dataset = file1.readlines() + file2.readlines()
    random.shuffle(combined_raw_dataset)
    line_number = 0
    for line in combined_raw_dataset:
        position_eval = line.split(",")
        if 'w' == position_eval[0].split()[1]:
            normalized_evaluation = 1 if "#" in position_eval[1] else 2 * (1 / (1 + math.exp(-float(position_eval[1])/300)) - 0.5) # normalizes centipawn score with sigmoid function
            position_eval[0] = position_eval[0] + (input_length - len(position_eval[0])) * " " # Add padding on FEN
            position_array = np.asarray([i.split(" ") for i in str(chess.Board(position_eval[0])).split("\n")])
            position_array_int = np.zeros((8, 8, 12))
            for i in range(0, len(position_array)):
                for j in range(0, len(position_array[i])):
                    position_array_int[i][j] = piece_char_2_int[position_array[i][j]]

            print(line_number)
            # print(position_eval)
            # print(chess.Board(position_eval[0]).unicode())
            # print(chess.Board(position_eval[0]))
            # print(position_array_int)
            # print("CentiPawn Score:", position_eval[1])
            # print("Normalized Evaluation:", normalized_evaluation)
            # print("Color:", color)
            positions[line_number] = position_array_int
            evaluations[line_number] = normalized_evaluation


            line_number += 1
            if line_number == wanted_dataset_size:
                break

if "yes" in input("\n\nWould you like to save (yes or no)"):
    save_name = input("\nSave as:")
    np.save("evaluations_" + save_name, np.asarray(evaluations))
    np.save("positions_" + save_name, np.asarray(positions))
    # np.save("colors_" + save_name, np.asarray(colors))


