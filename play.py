import numpy as np
import copy
import chess
import tensorflow as tf
import time

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


model = tf.keras.models.load_model("chess_engine_v1.h5")

while True:
    max_evaluation_score = -1
    max_evaluation_score_move = ""
    legal_moves = [board.san(move) for move in list(board.legal_moves)]
    print(legal_moves)
    for alg_move in legal_moves:
        board_copy = chess.Board(board.fen())
        board_copy.push_san(alg_move)
        letter_position = np.asarray([i.split(" ") for i in str(board).split("\n")]) # gives position using [r, R, k, K, b, B, q, Q, k, K, p, P, .] notation
        one_hot_board = np.zeros((8, 8, 12))
        for i in range(0, len(letter_position)):
            for j in range(0, len(letter_position[i])):
                one_hot_board[i][j] = piece_char_2_int[letter_position[i][j]]
        evaluation_score = model.predict(np.expand_dims(one_hot_board,0))
        if evaluation_score >= max_evaluation_score:
            max_evaluation_score_move = alg_move
            max_evaluation_score = evaluation_score
    time.sleep(1)

    board.push_san(max_evaluation_score_move)
    print("\nComputer making move: %s with evaluation score %0.3f\n" % (max_evaluation_score_move, max_evaluation_score))
    print(board.unicode())

    legal_moves = [board.san(move) for move in list(board.legal_moves)]
    print("Playing as %s " % str("White" if board.turn else "Black"))
    print("Legal Moves: %s" % str(legal_moves))
    wanted_player_move = input("Enter your move: " )
    while legal_moves.count(wanted_player_move) == 0:
        wanted_player_move = input("\nUnavailable move or incorrect algebraic chess notation. Please try entering correct legal move \n [%s]: " % str(legal_moves))
    board.push_san(wanted_player_move)
    print(board.unicode())
    time.sleep(1)

