import chess
board = chess.Board()

print(board.legal_moves)

board.push_san("e4")


print(board.legal_moves)
