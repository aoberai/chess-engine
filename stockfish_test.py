import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/home/aoberai/programming/python/chess-engine/stockfish")

board = chess.Board()
limit = chess.engine.Limit(time=2.0)
# print(engine.play(board, limit).move)
print(board.unicode())
print(engine.analyse(board, chess.engine.Limit(time=0.4))["score"].white().score()) # normalizes centipawn score with sigmoid function

board.push_san("e4")
print(board.unicode())
print(engine.analyse(board, chess.engine.Limit(time=0.4))["score"].white().score()) # normalizes centipawn score with sigmoid function


board.push_san("a5")
print(board.unicode())
print(engine.analyse(board, chess.engine.Limit(time=0.4))["score"].white().score()) # normalizes centipawn score with sigmoid function


engine.quit()
