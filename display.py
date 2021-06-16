
import time
import chess
import chess.svg
from flask import Flask, Markup, render_template
from flask_table import Table, Col
import base64
board = chess.Board()
board.push_san("e4")

print(board)
#
# class ChessGrid(Table):
#     a = Col(" ")
#     b = Col(" ")
#     c = Col(" ")
#     d = Col(" ")
#     e = Col(" ")
#     f = Col(" ")
#     g = Col(" ")
#     h = Col(" ")
#
def to_svg(board):
    return base64.b64encode(chess.svg.board(board).encode('utf-8')).decode('utf-8')

app = Flask(__name__)
@app.route("/")
def home():
    board_svg = to_svg(board)
    ret = '<html><head>'
    ret += '<style>input { font-size: 30px; } button { font-size: 30px; }</style>'
    ret += '</head><body>'
    ret += '<img width=1000 height=1000 src="data:image/svg+xml;base64,%s"></img><br/>' % board_svg
    ret += '<form action="/move"><input name="move" type="text"></input><input type="submit" value="Move"></form><br/>'
    return ret


@app.route("/selfplay")
def selfplay():
    ret = '<html><head>'
    ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(board)
    return ret
#
# app.run(debug=True)

# move given in algebraic notation
@app.route("/move")
def move():
    board.push_san("e5")
    response = app.response_class(
        response=board.fen(),
        status=200
    )
    # return response
    return home()

if __name__ == "__main__":
    app.run(debug=True)














# @app.route('/')
# def home():
#     board_list = board.unicode().replace("\n", "<br>")
#     # temp = board_list.split("<br>")
#     # gridContent = {}
#     # gridContent['a'] = temp[0]
#     # gridContent['b'] = temp[1]
#     # gridContent['c'] = temp[2]
#     # gridContent['d'] = temp[3]
#     # gridContent['e'] = temp[4]
#     # gridContent['f'] = temp[5]
#     # gridContent['g'] = temp[6]
#     # gridContent['h'] = temp[7]
#     #
#     # # print(board_list.replace("<br>", "\n"))
#     # # return board_list
#     # table = ChessGrid(gridContent)
#     # table.border = True
#     return render_template('index.html', board=board_list)
# # return render_template('index.html', chess_table=table)


