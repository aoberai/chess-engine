import numpy as np
import math

# board = "r n . q k b . r\n. . . . p p . p\n. P . . . n p .\np . p P p . . .\n. . . . . . . .\n. . N . . N . .\nP P . . . P P P\nR . B Q . K . R"

normalized_evaluation = 2 * (1 / (1 + math.exp(-float(130)/300)) - 0.5)
print(normalized_evaluation)
normalized_evaluation = 2 * (1 / (1 + math.exp(-float(-500)/300)) - 0.5)

print(normalized_evaluation)


# print(board)
# print(np.asarray([i.split(" ") for i in board.split("\n")]))


piece_char_2_int = {"R" : 21, "N": 22, "B": 23, "Q" : 24, "K" : 25, "P" : 26, "r" : 11, "n": 12, "b": 13, "q" : 14, "k" : 15, "p" : 16, "." : 0} # Uppercase is Black, lowercase white. Conversion to int since model can only take float input, not string

inv_map = {v: k for k, v in piece_char_2_int.items()}

print(inv_map)
position = [[11,  0,  0, 14,  0, 11, 15,  0], [ 0, 16, 16,  0,  0, 16, 16, 16,], [16,  0, 12, 16, 13, 12,  0,  0], [ 0,  0, 13,  0, 16,  0,  0,  0], [ 0,  0,  0,  0, 26,  0,  0,  0], [ 0, 23, 26, 26,  0, 22,  0, 26],  [26, 26,  0, 22,  0, 26, 26,  0], [21,  0, 23, 24, 25,  0,  0, 21]]


for i in range(0, len(position)):
    for j in range(0, len(position[i])):
        print(" ", inv_map[position[i][j]], end = ' ')
    print("\n")



