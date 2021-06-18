import numpy as np
import math
import chess


# def position_evaluation(fen):
#     board_copy = chess.Board(fen)
#     one_hot_board = serialize_position(board_copy)
#     evaluation_score = model.predict(one_hot_board)[0][0]
#     return evaluation_score

#recursive search of all available moves from given point
def minimax(fen, depth=1): # depth represents ply
    # Establish Search Tree

    board = chess.Board(fen)

    if depth == 0:
        print(board.unicode())
        return fen
    legal_moves = [board.san(move) for move in list(board.legal_moves)]
    for move in legal_moves:
        next_board = chess.Board(fen)
        next_board.push_san(move)
        minimax(next_board.fen(), depth - 1)




(minimax(chess.Board().fen()))



#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats as stats
# import math
#
# mu = 0
# variance = 1
# sigma = math.sqrt(variance)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, 1/stats.norm.pdf(x, mu, sigma))
# plt.show()
# # evaluations = np.load("evaluations_W750K.npy")
# # num_bins = 20
# # n, bins, patches = plt.hist(evaluations.tolist(), num_bins, facecolor='blue', alpha=0.5)
# #
# # plt.axvline(evaluations.mean(), color='k', linestyle='dashed', linewidth=1)
# # plt.axvline(np.median(evaluations), color='k', linestyle='dashed', linewidth=1)
# #
# # print("Mean: %0.4f" % evaluations.mean())
# # print("Median: %0.4f" % np.median(evaluations))
# # plt.show()
# #
#
# # # board = "r n , q k b . r\n. . . . p p . p\n. P . . . n p .\np . p P p . . .\n. . . . . . . .\n. . N . . N . .\nP P . . . P P P\nR . B Q . K . R"
# #
# # normalized_evaluation = 2 * (1 / (1 + math.exp(-float(130)/300)) - 0.5)
# # print(normalized_evaluation)
# # normalized_evaluation = 2 * (1 / (1 + math.exp(-float(-500)/300)) - 0.5)
# #
# # print(normalized_evaluation)
# #
# #
# # # print(board)
# # # print(np.asarray([i.split(" ") for i in board.split("\n")]))
# #
# #
# # # piece_char_2_int = {"R" : 21, "N": 22, "B": 23, "Q" : 24, "K" : 25, "P" : 26, "r" : 11, "n": 12, "b": 13, "q" : 14, "k" : 15, "p" : 16, "." : 0 # Uppercase is Black, lowercase white. Conversion to int since model can only take float input, not string
# # #
# # piece_char_2_int = {
# #         'p' : (1,0,0,0,0,0,0,0,0,0,0,0),
# #         'P' : (0,0,0,0,0,0,1,0,0,0,0,0),
# #         'n' : (0,1,0,0,0,0,0,0,0,0,0,0),
# #         'N' : (0,0,0,0,0,0,0,1,0,0,0,0),
# #         'b' : (0,0,1,0,0,0,0,0,0,0,0,0),
# #         'B' : (0,0,0,0,0,0,0,0,1,0,0,0),
# #         'r' : (0,0,0,1,0,0,0,0,0,0,0,0),
# #         'R' : (0,0,0,0,0,0,0,0,0,1,0,0),
# #         'q' : (0,0,0,0,1,0,0,0,0,0,0,0),
# #         'Q' : (0,0,0,0,0,0,0,0,0,0,1,0),
# #         'k' : (0,0,0,0,0,1,0,0,0,0,0,0),
# #         'K' : (0,0,0,0,0,0,0,0,0,0,0,1),
# #         '.' : (0,0,0,0,0,0,0,0,0,0,0,0),
# # }
# #
# #
# # inv_map = {v: k for k, v in piece_char_2_int.items()}
# #
# # print(inv_map)
# # position = [[(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
# #     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #     (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
# #
# #   [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
# #
# #   [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
# #
# #   [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
# #
# #   [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
# #
# #   [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
# #
# #   [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)],
# #
# #   [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
# #      (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]]
# #
# # for i in range(0, len(position)):
# #     for j in range(0, len(position[i])):
# #         print(" ", inv_map[position[i][j]], end = ' ')
# #     print("\n")
