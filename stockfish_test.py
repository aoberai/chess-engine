import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/home/aoberai/programming/python/chess-engine/stockfish")

board = chess.Board()
limit = chess.engine.Limit(time=2.0)
print(engine.play(board, limit).move)

engine.quit()
