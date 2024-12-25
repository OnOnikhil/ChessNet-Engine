import React, { useState } from 'react';

const ChessAI = () => {
  const [fen, setFen] = useState('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const [predictedMove, setPredictedMove] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Use same triangular symbol for all pawns, same symbols for other pieces
  const pieceSymbols = {
    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♙',  // black pieces (lowercase)
    'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♙'   // white pieces (uppercase)
  };


  const renderChessboard = () => {
    const rows = fen.split(' ')[0].split('/');
    const boardRows = [];

    for (let row of rows) {
      const cells = [];
      for (const char of row) {
        if (isNaN(char)) {
          // For pieces, store the character directly
          cells.push(char);
        } else {
          // For empty squares, add that many null values
          for (let i = 0; i < parseInt(char); i++) {
            cells.push(null);
          }
        }
      }
      boardRows.push(cells);
    }

    return (
      <div className="relative w-[560px] h-[560px]">
        {/* File labels (a-h) */}
        <div className="absolute -bottom-6 left-0 right-0 flex justify-around text-gray-400">
          {['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'].map(file => (
            <div key={file} className="w-[70px] text-center">{file}</div>
          ))}
        </div>
        
        {/* Rank labels (1-8) */}
        <div className="absolute -left-6 top-0 bottom-0 flex flex-col justify-around text-gray-400">
          {[8, 7, 6, 5, 4, 3, 2, 1].map(rank => (
            <div key={rank} className="h-[70px] flex items-center">{rank}</div>
          ))}
        </div>
        
        <div className="w-full h-full grid grid-cols-8 border border-gray-600">
          {boardRows.map((row, rowIndex) => (
            row.map((piece, colIndex) => {
              const isLight = (rowIndex + colIndex) % 2 === 0;
              
              return (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className={`
                    w-[70px] h-[70px]
                    flex items-center justify-center
                    ${isLight ? 'bg-[#ebd7b8]' : 'bg-[#b78b62]'}
                    text-4xl
                  `}
                >
                  {piece && (
                    <span className={`
                      ${piece === piece.toLowerCase() ? 'text-gray-800' : 'text-white'}
                      font-bold
                    `}>
                      {pieceSymbols[piece]}
                    </span>
                  )}
                </div>
              );
            })
          ))}
        </div>
      </div>
    );
  };



const getPrediction = async () => {
    setLoading(true);
    setError('');
    console.log('Sending FEN:', fen);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen }),
      });
      console.log('Response status:', response.status);
      const text = await response.text();
      console.log('Response text:', text);
  
      if (!response.ok) {
        throw new Error(`Server error: ${text}`);
      }
  
      const data = JSON.parse(text);
      setPredictedMove(data.move);
    } catch (err) {
      console.error('Fetch error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#1b1e23] py-8">
      <div className="max-w-4xl mx-auto px-4">
        <div className="bg-gray-900 rounded-lg shadow-lg overflow-hidden">
          <div className="p-6">
            <h1 className="text-2xl font-bold text-white mb-6">
              Chess AI Move Predictor
            </h1>
            
            <div className="space-y-6">
              <div className="flex justify-center">
                {renderChessboard()}
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Enter FEN Position:
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={fen}
                      onChange={(e) => setFen(e.target.value)}
                      className="flex-1 min-w-0 rounded-md border border-gray-600 bg-gray-800 px-3 py-2 text-white placeholder-gray-400"
                      placeholder="Enter FEN notation..."
                    />
                    <button
                      onClick={getPrediction}
                      disabled={loading}
                      className={`
                        px-4 py-2 rounded-md text-white font-medium
                        ${loading 
                          ? 'bg-blue-600 cursor-not-allowed' 
                          : 'bg-blue-500 hover:bg-blue-600'}
                        transition-colors
                      `}
                    >
                      {loading ? 'Analyzing...' : 'Get Best Move'}
                    </button>
                  </div>
                </div>

                {error && (
                  <div className="rounded-md bg-red-900/50 p-4">
                    <div className="text-sm text-red-200">
                      {error}
                    </div>
                  </div>
                )}

                {predictedMove && (
                  <div className="rounded-md bg-green-900/50 p-4">
                    <h3 className="text-sm font-medium text-green-200">
                      Predicted Best Move:
                    </h3>
                    <p className="mt-2 text-sm text-green-100 font-mono">
                      {predictedMove}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChessAI;