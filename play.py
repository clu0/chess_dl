import traceback
from flask import Flask, Response, request
import chess
import chess.svg
import base64
from mcts import MCTSChess, FakeChessNet


app = Flask(__name__)
s = chess.Board()
net = FakeChessNet(depth=10)
agent = MCTSChess(net, 1)

def to_svg(s):
    return base64.b64encode(chess.svg.board(board=s).encode('utf-8')).decode('utf-8')

@app.route("/move")
def move():
    if not s.is_game_over():
        user_mv = request.args.get('move',default="")
    if user_mv is not None and user_mv != "":
        print("human moves", user_mv)
        try:
            s.push_san(user_mv)
            if s.is_game_over():
                print("GAME IS OVER")
            mv = agent.make_move(s, n_sim=10)
            s.push(mv)
        except Exception:
            traceback.print_exc()
    else:
        print("GAME IS OVER")
    return hello()

@app.route("/")
def hello():
    board_svg = to_svg(s)
    ret = '<html><head>'
    ret += '<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>'
    ret += '<style>input { font-size: 30px; } button { font-size: 30px; }</style>'
    ret += '</head><body>'
    ret += '<a href="/reset">New game</a><br/>'
    ret += '<a href="/play_black">New game as black</a><br/>'
    ret += '<a href="/undo">Undo last move</a><br/>'
    ret += '<img width=600 height=600 src="data:image/svg+xml;base64,%s"></img><br/>' % board_svg
    ret += '<form action="/move"><input id="move" name="move" type="text"></input><input type="submit" value="Move"></form><br/>'
    ret += '<script>$(function() { var input = document.getElementById("move"); console.log("selected"); input.focus(); input.select(); }); </script>'
    return ret

    
@app.route("/undo")
def undo():
    _ = s.pop()
    return hello()

@app.route("/reset")
def reset():
    s.reset_board()
    return hello()

@app.route("/play_black")
def play_black():
    s.reset_board()
    mv = agent.make_move(s, n_sim=10)
    s.push(mv)
    return hello()
    
    

#@app.route("/board.svg")
#def board():
#    return Response(chess.svg.board(board=chess.Board()), mimetype="image/svg+xml")

if __name__ == "__main__":
    app.run(debug=True)
