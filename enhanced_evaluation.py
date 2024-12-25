import chess
import chess.engine
import torch
from model import ChessNet, board_2_rep, get_device
import numpy as np
from tqdm import tqdm

class EnhancedEvaluator:
    def __init__(self, model_path='chess_model.pth'):
        self.device = get_device()
        self.model = ChessNet().to(self.device)  # Move model to device
        
        # Load and move weights to device
        weights = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(weights)
        self.model.eval()
        
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        except Exception as e:
            print(f"Error initializing Stockfish: {e}")
            print("Make sure Stockfish is installed and in your PATH")
            self.engine = None

    def get_model_move(self, fen):
        """Get move prediction from your model with improved debugging"""
        board = chess.Board(fen)
        board_tensor = torch.FloatTensor(board_2_rep(board)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(board_tensor)
        
        # Get probabilities
        from_probs = torch.softmax(output[0, 0].view(-1), dim=0).cpu().numpy()
        to_probs = torch.softmax(output[0, 1].view(-1), dim=0).cpu().numpy()
        
        print("\nAnalyzing Model's Decision:")
        print("Top 5 'From' Square Probabilities:")
        from_indices = np.argsort(from_probs)[-5:][::-1]
        for idx in from_indices:
            rank = idx // 8
            file = idx % 8
            square = chess.square(file, 7-rank)  # Adjust coordinates
            print(f"{chess.square_name(square)}: {from_probs[idx]:.4f}")
            # Print pieces at these squares
            piece = board.piece_at(square)
            print(f"Piece at {chess.square_name(square)}: {piece if piece else 'Empty'}")

        print("\nTop 5 'To' Square Probabilities:")
        to_indices = np.argsort(to_probs)[-5:][::-1]
        for idx in to_indices:
            rank = idx // 8
            file = idx % 8
            square = chess.square(file, 7-rank)  # Adjust coordinates
            print(f"{chess.square_name(square)}: {to_probs[idx]:.4f}")

        # Try to find a legal move from the top predictions
        for from_idx in from_indices:
            from_rank = from_idx // 8
            from_file = from_idx % 8
            from_square = chess.square(from_file, 7-from_rank)
            
            for to_idx in to_indices:
                to_rank = to_idx // 8
                to_file = to_idx % 8
                to_square = chess.square(to_file, 7-to_rank)
                
                move = chess.Move(from_square, to_square)
                if move in board.legal_moves:
                    print(f"Found legal move: {move.uci()}")
                    return move

        # If no legal move found from top predictions, fall back to first legal move
        print("No legal move found from top predictions, falling back to first legal move")
        return next(iter(board.legal_moves))
    
    def calculate_move_similarity(self, move1, move2):
        """Calculate similarity between two moves"""
        board = chess.Board()
        from1 = chess.parse_square(move1[:2])
        to1 = chess.parse_square(move1[2:])
        from2 = chess.parse_square(move2[:2])
        to2 = chess.parse_square(move2[2:])
        
        # Calculate Manhattan distance between squares
        from_dist = abs((from1 % 8) - (from2 % 8)) + abs((from1 // 8) - (from2 // 8))
        to_dist = abs((to1 % 8) - (to2 % 8)) + abs((to1 // 8) - (to2 // 8))
        
        # Normalize distances (max distance on board is 14)
        similarity = 1 - ((from_dist + to_dist) / 28)
        return similarity
    
    def evaluate_position_detailed(self, fen, depth=20):
        """Detailed evaluation of a position"""
        board = chess.Board(fen)
        print(f"\nEvaluating position:\n{board}")
        
        # Get model's move
        model_move = self.get_model_move(fen)
        
        # Get Stockfish's top 3 moves
        try:
            multipv_analysis = self.engine.analyse(
                board,
                chess.engine.Limit(depth=depth, time=0.1),  # Added time limit
                multipv=3
            )
            
            stockfish_moves = [info["pv"][0] for info in multipv_analysis]
            stockfish_scores = []
            for info in multipv_analysis:
                try:
                    score = info["score"].relative.score()
                    if score is not None:
                        stockfish_scores.append(score)
                    else:
                        stockfish_scores.append(0)  # Default score if None
                except Exception as e:
                    print(f"Error getting score: {e}")
                    stockfish_scores.append(0)
            
        except Exception as e:
            print(f"Error in Stockfish analysis: {e}")
            stockfish_moves = [next(iter(board.legal_moves))]
            stockfish_scores = [0]
        
        # Calculate move similarities
        similarities = [self.calculate_move_similarity(
            model_move.uci(), 
            sf_move.uci()
        ) for sf_move in stockfish_moves]
        
        # Evaluate position after model's move
        try:
            board.push(model_move)
            if not board.is_game_over():
                try:
                    model_position = self.engine.analyse(
                        board, 
                        chess.engine.Limit(depth=depth, time=0.1)
                    )
                    score = model_position["score"].relative.score()
                    model_score = score if score is not None else 0
                except Exception as e:
                    print(f"Error analyzing position after move: {e}")
                    model_score = 0
            else:
                model_score = 0
            board.pop()
        except Exception as e:
            print(f"Error evaluating model move: {e}")
            model_score = 0
        
        return {
            "position": fen,
            "model_move": model_move.uci(),
            "stockfish_moves": [m.uci() for m in stockfish_moves],
            "stockfish_scores": stockfish_scores,
            "move_similarities": similarities,
            "position_score": model_score,
            "evaluation_difference": (
                abs(model_score - stockfish_scores[0])
                if model_score is not None and stockfish_scores and stockfish_scores[0] is not None
                else None
            )
        }
    
    def evaluate_strategic_understanding(self, num_positions=1):
        """Evaluate model's strategic understanding"""
        test_positions = {
            "opening": ["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"],
            "middlegame": ["r2qk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w kq - 0 8"],
            "endgame": ["4k3/4P3/8/8/8/8/4K3/8 w - - 0 1"],
            "tactical": ["r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"]
        }
        
        results = []
        for category, positions in test_positions.items():
            category_results = []
            for fen in positions[:num_positions]:
                print(f"\nEvaluating {category} position...")
                eval_result = self.evaluate_position_detailed(fen)
                category_results.append(eval_result)
            
            avg_similarity = np.mean([r["move_similarities"][0] for r in category_results])
            avg_eval_diff = np.mean([r["evaluation_difference"] for r in category_results if r["evaluation_difference"] is not None])
            
            results.append({
                "category": category,
                "results": category_results,
                "average_similarity": avg_similarity,
                "average_eval_diff": avg_eval_diff
            })
        
        return results

    def print_detailed_report(self, strategic_results):
        print("\nDetailed Strategic Analysis:")
        print("=" * 50)
        
        for category in strategic_results:
            print(f"\n{category['category'].upper()} Analysis:")
            print(f"Average move similarity to Stockfish: {category['average_similarity']:.2%}")
            print(f"Average evaluation difference: {category['average_eval_diff']:.1f}")
            
            for result in category['results']:
                print(f"\nPosition: {result['position']}")
                print(f"Model move: {result['model_move']}")
                print(f"Stockfish's top moves: {', '.join(result['stockfish_moves'])}")
                print(f"Move similarity: {result['move_similarities'][0]:.2%}")
                if result['evaluation_difference'] is not None:
                    print(f"Evaluation difference: {result['evaluation_difference']:.1f}")

    def __del__(self):
        if hasattr(self, 'engine') and self.engine:
            self.engine.quit()

if __name__ == "__main__":
    print("Starting enhanced model evaluation...")
    evaluator = EnhancedEvaluator()
    strategic_results = evaluator.evaluate_strategic_understanding()
    evaluator.print_detailed_report(strategic_results)