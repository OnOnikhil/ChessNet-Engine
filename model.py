import numpy as np
import re
import pandas as pd
import gc
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Device configuration
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

device = get_device()
print(f"Using device: {device}")

# Your existing mappings
letter_2_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
num_2_letter = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}

# Board representation functions
def board_2_rep(board):
    pieces = ['p', 'r', 'n','b','q','k']
    layers = []
    for piece in pieces:
        layers.append(create_rep_layer(board, piece))
    board_rep = np.stack(layers)
    return board_rep

def create_rep_layer(board, type):
    s = str(board)
    s = re.sub(f'[^{type}{type.upper()} \n]', '.', s)
    s = re.sub(f'{type}', '-1', s)
    s = re.sub(f'{type.upper()}','1', s)
    s = re.sub(f'\.', '0', s)

    board_mat = []
    for row in s.split('\n'):
        row = row.split(' ')
        row = [int(x) for x in row]
        board_mat.append(row)
        
    return np.array(board_mat)

def move_2_rep(move, board):
    board.push_san(move).uci()
    move = str(board.pop())

    from_output_layer = np.zeros((8,8))
    from_row = 8 - int(move[1])
    from_column = letter_2_num[move[0]]
    from_output_layer[from_row, from_column] = 1

    to_output_layer = np.zeros((8,8))
    to_row = 8 - int(move[3])
    to_column = letter_2_num[move[2]]
    to_output_layer[to_row, to_column] = 1

    return np.stack([from_output_layer, to_output_layer])

def create_move_list(s):
    return re.sub('\d*\. ','',s).split(' ')[:-1]

# Data augmentation functions
def rotate_board(board_rep, k=1):
    return np.array(np.rot90(board_rep, k=k, axes=(1, 2)), copy=True)

def flip_board(board_rep, axis=1):
    return np.array(np.flip(board_rep, axis=axis), copy=True)

# Endgame dataset creation
def create_endgame_dataset():
    endgame_positions = [
        "4k3/4P3/4K3/8/8/8/8/8 w - - 0 1",  # King and pawn vs king
        "4k3/8/4K3/8/8/8/8/R7 w - - 0 1",   # Rook and king vs king
        "8/4k3/8/8/8/3K4/8/R7 w - - 0 1",   # Another rook endgame
        "8/5k2/8/8/8/3K4/8/Q7 w - - 0 1",   # Queen endgame
        "8/8/8/8/5k2/3K4/8/B7 w - - 0 1",   # Bishop endgame
        # Add more common endgame positions
    ]
    
    boards = []
    moves = []
    
    for fen in endgame_positions:
        board = chess.Board(fen)
        # Generate optimal moves for these positions using minimax or tablebase
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            boards.append(board.copy())
            moves.append(move)
    
    return boards, moves




class ChessDataset(Dataset):
    def __init__(self, data, include_endgames=False):
        self.data = data
        self.include_endgames = include_endgames
        # Add initialization for endgame boards and moves if required

    def __len__(self):
        return len(self.data)  # Replace with the actual size of your dataset

    def __getitem__(self, index):
        # Your logic to fetch and preprocess data
        moves = []

        # Example logic to handle both normal and endgame scenarios
        if self.include_endgames and torch.rand(1).item() < 0.2:
            idx = torch.randint(len(self.endgame_boards), (1,)).item()
            board = self.endgame_boards[idx]
            next_move = self.endgame_moves[idx]
        else:
            game_i = torch.randint(self.data.shape[0], (1,)).item()
            random_game = self.data['AN'].values[game_i]
            moves = create_move_list(random_game)
            game_state_i = torch.randint(len(moves) - 1, (1,)).item()
            next_move = moves[game_state_i]
            moves = moves[:game_state_i]
            board = chess.Board()
            for move in moves:
                board.push_san(move)

        x = board_2_rep(board)
        y = move_2_rep(str(next_move), board)

        # Data augmentation (optional)
        if torch.rand(1).item() < 0.5:
            k = torch.randint(4, (1,)).item()
            x = rotate_board(x, k)
            y = rotate_board(y, k)

        if torch.rand(1).item() < 0.5:
            axis = torch.randint(2, (1,)).item() + 1
            x = flip_board(x, axis)
            y = flip_board(y, axis)

        if len(moves) % 2 == 1:
            x *= -1

        return torch.FloatTensor(x), torch.FloatTensor(y)



class module(nn.Module):
    def __init__(self, hidden_size):
        super(module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.dropout = nn.Dropout2d(0.1)
        self.activation1 = nn.SELU()
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input
        x = self.activation2(x)
        return x

class ChessNet(nn.Module):
    def __init__(self, hidden_layers=8, hidden_size=256):
        super(ChessNet, self).__init__()
        
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_size)
        
        self.module_list = nn.ModuleList([module(hidden_size) for _ in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = F.relu(x)

        for module in self.module_list:
            x = module(x)
            
        x = self.output_layer(x)
        return x

# Training and model evaluation functions remain the same...
# Place the train_model function definition above the main function

def train_model(model, train_loader, num_epochs=10):
    print("Entering train_model function...")
    model.train()
    print("Model set to train mode")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    print("Optimizer created")
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    print("Scheduler created")
    
    # Change to MSE Loss since we're predicting continuous values
    criterion = nn.MSELoss()
    print("Loss function initialized")
    
    print(f"Starting training on {device}...")
    print(f"Total epochs: {num_epochs}")
    print(f"Total batches per epoch: {len(train_loader)}")
    
    try:
        for epoch in range(num_epochs):
            print(f"\nStarting epoch {epoch+1}/{num_epochs}")
            total_loss = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                if batch_idx == 0:
                    print(f"First batch shapes - x: {x.shape}, y: {y.shape}")
                
                x = x.to(device)
                y = y.to(device)
                
                optimizer.zero_grad()
                output = model(x)  # Shape: [batch_size, 2, 8, 8]
                
                # Calculate loss directly without reshaping
                loss = criterion(output, y)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

    print("Training completed!")

def main():
    # Load and preprocess data
    print("Loading chess data...")
    chess_data_raw = pd.read_csv('chess_games.csv', usecols=['AN', 'WhiteElo'])
    chess_data = chess_data_raw[chess_data_raw['WhiteElo'] > 2000]
    del chess_data_raw
    gc.collect()
    chess_data = chess_data[['AN']]
    chess_data = chess_data[~chess_data['AN'].str.contains('{')]
    chess_data = chess_data[chess_data['AN'].str.len() > 20]
    print(f"Loaded {len(chess_data)} games")

    # Create dataset and dataloader
    print("Creating dataset...")
    data_train = ChessDataset(chess_data)
    print("Creating dataloader...")
    data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, drop_last=True, num_workers=2)
    print("Dataloader created")

    # Initialize model
    print("Initializing model...")
    model = ChessNet()
    model = model.to(device)
    print("Model initialized and moved to device")
    
    # Train model
    print("Starting training process...")
    train_model(model, data_train_loader)
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), 'chess_model.pth')
    print("Model saved to chess_model.pth")

if __name__ == "__main__":
    main()
