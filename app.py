# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import torch
from model import ChessNet, board_2_rep, get_device
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return "Chess AI Server is Running!"

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

device = get_device()
model = ChessNet()
try:
    model.load_state_dict(torch.load('chess_model.pth', map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def get_best_move(board_fen):
    board = chess.Board(board_fen)
    board_tensor = torch.FloatTensor(board_2_rep(board)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(board_tensor)
    
    from_probs = torch.softmax(output[0, 0].flatten(), dim=0).cpu().numpy()
    to_probs = torch.softmax(output[0, 1].flatten(), dim=0).cpu().numpy()
    
    from_idx = np.argmax(from_probs)
    to_idx = np.argmax(to_probs)
    
    from_file = from_idx % 8
    from_rank = from_idx // 8
    to_file = to_idx % 8
    to_rank = to_idx // 8
    
    from_sq = chess.square(from_file, 7-from_rank)
    to_sq = chess.square(to_file, 7-to_rank)
    
    move = chess.Move(from_sq, to_sq)
    if move in board.legal_moves:
        return move.uci()
    
    return next(iter(board.legal_moves)).uci()

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})
        
    try:
        fen = request.json['fen']
        move = get_best_move(fen)
        return jsonify({'move': move})
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)