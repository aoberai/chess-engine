import chess
from flask import Flask, Markup, render_template

board = chess.Board()
board.push_san("e4")

print(board)

app = Flask(__name__)

@app.route('/')
def home():
    # print(chess.svg.board(board))
    return render_template('index.html', svg=board.unicode())

app.run()


