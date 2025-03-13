import axios from "axios";
import { useEffect, useState } from "react";
import Slot from "./Slot";

const Board = () => {
    const [board, setBoard] = useState([
        ['', '', '', '', '', '', ''],
        ['', '', '', '', '', '', ''],
        ['', '', '', '', '', '', ''],
        ['', '', '', '', '', '', ''],
        ['', '', '', '', '', '', ''],
        ['', '', '', '', '', '', ''],
    ]);
    const [currPlayer, setCurrPlayer] = useState('X');
    const [oppPlayer, setOppPlayer] = useState('O');
    const [gameOver, setGameOver] = useState(false);
    const [difficulty, setDifficulty] = useState('medium');

    const handleDifficultyClick = (e) => {
        setDifficulty(e.target.value);
    }

    //run only when its 'O' aka model's move
    useEffect(() => {
        if(currPlayer==='O' && !gameOver) {
            getModelMove(board);
        }
    }, [currPlayer]); //runs whenever 'currPlayer' updates

    
    const checkWin = (row, column, ch) => {
        //down 4
        try {
            if (board[row + 1][column] === ch) {
                if (board[row + 2][column] === ch) {
                    if (board[row + 3][column] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) { console.log(e) }

        //side right 4
        try {
            if (board[row][column + 1] === ch) {
                if (board[row][column + 2] === ch) {
                    if (board[row][column + 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) { console.log(e) }

        //side left 4
        try {
            if (board[row][column - 1] === ch) {
                if (board[row][column - 2] === ch) {
                    if (board[row][column - 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) { console.log(e) }

        //diagonal down right 4 
        try {
            if (board[row + 1][column + 1] === ch) {
                if (board[row + 2][column + 2] === ch) {
                    if (board[row + 3][column + 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) { console.log(e) }

        //diagonal down left 4 
        try {
            if (board[row + 1][column - 1] === ch) {
                if (board[row + 2][column - 2] === ch) {
                    if (board[row + 3][column - 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) { console.log(e) }

        //diagonal upper left 4
        try {
            if (board[row - 1][column - 1] === ch) {
                if (board[row - 2][column - 2] === ch) {
                    if (board[row - 3][column - 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) { console.log(e) }

        // diagonal upper right 4 
        try {
            if (board[row - 1][column + 1] === ch) {
                if (board[row - 2][column + 2] === ch) {
                    if (board[row - 3][column + 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) { console.log(e) }
    }

    /*
    const checkWin = (row, column, ch) => {
        const directions = [
            { rowOffset: -1, colOffset: 0 },   // down
            { rowOffset: 0, colOffset: 1 },    // right
            { rowOffset: 0, colOffset: -1 },   // left
            { rowOffset: 1, colOffset: 1 },    // down-right (diagonal)
            { rowOffset: 1, colOffset: -1 },   // down-left (diagonal)
            { rowOffset: -1, colOffset: -1 },  // up-left (diagonal)
            { rowOffset: -1, colOffset: 1 },   // up-right (diagonal)
        ];
    
        for (let dir of directions) {
            let count = 1; // count includes the current piece
    
            // Check in one direction
            for (let i = 1; i < 4; i++) {
                const newRow = row + dir.rowOffset * i;
                const newCol = column + dir.colOffset * i;
    
                // Ensure we don't go out of bounds and check for matching tokens
                if (newRow >= 0 && newRow < board.length && newCol >= 0 && newCol < board[0].length) {
                    if (board[newRow][newCol] === ch) {
                        count++;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
    
            // Check in the opposite direction
            for (let i = 1; i < 4; i++) {
                const newRow = row - dir.rowOffset * i;
                const newCol = column - dir.colOffset * i;
    
                // Ensure we don't go out of bounds and check for matching tokens
                if (newRow >= 0 && newRow < board.length && newCol >= 0 && newCol < board[0].length) {
                    if (board[newRow][newCol] === ch) {
                        count++;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
    
            // If a line of 4 is found (including the current piece)
            if (count >= 4) {
                return true;
            }
        }
    
        return false;
    };*/

    /* online checkWin func logic
    const checkWin = (ch) => {
        for (let i = 0; i < board.length; i++) {
            for (let j = 0; j < board[0].length; j++) {
                if (board[i][j] === ch) {
                    // Downwards
                    try {
                        if (i + 3 < board.length && 
                            board[i + 1][j] === ch && 
                            board[i + 2][j] === ch && 
                            board[i + 3][j] === ch) {
                            return true;
                        }
                    } catch (e) {}
    
                    // Down-right diagonal
                    try {
                        if (i + 3 < board.length && j + 3 < board[0].length &&
                            board[i + 1][j + 1] === ch && 
                            board[i + 2][j + 2] === ch && 
                            board[i + 3][j + 3] === ch) {
                            return true;
                        }
                    } catch (e) {}
    
                    // Down-left diagonal
                    try {
                        if (i + 3 < board.length && j - 3 >= 0 &&
                            board[i + 1][j - 1] === ch && 
                            board[i + 2][j - 2] === ch && 
                            board[i + 3][j - 3] === ch) {
                            return true;
                        }
                    } catch (e) {}
    
                    // Right
                    try {
                        if (j + 3 < board[0].length &&
                            board[i][j + 1] === ch && 
                            board[i][j + 2] === ch && 
                            board[i][j + 3] === ch) {
                            return true;
                        }
                    } catch (e) {}
    
                    // Left
                    try {
                        if (j - 3 >= 0 &&
                            board[i][j - 1] === ch && 
                            board[i][j - 2] === ch && 
                            board[i][j - 3] === ch) {
                            return true;
                        }
                    } catch (e) {}
    
                    // Up-right diagonal
                    try {
                        if (i - 3 >= 0 && j + 3 < board[0].length &&
                            board[i - 1][j + 1] === ch && 
                            board[i - 2][j + 2] === ch && 
                            board[i - 3][j + 3] === ch) {
                            return true;
                        }
                    } catch (e) {}
    
                    // Up-left diagonal
                    try {
                        if (i - 3 >= 0 && j - 3 >= 0 &&
                            board[i - 1][j - 1] === ch && 
                            board[i - 2][j - 2] === ch && 
                            board[i - 3][j - 3] === ch) {
                            return true;
                        }
                    } catch (e) {}
                }
            }
        }
        return false;
    }; */
    
    

    const updateBoard = (row, column, ch) => {
        setBoard(prev => {
            /*board = prev board + (X or O in (row, column) from the last move) */
            const boardCopy = [...prev];
            boardCopy[row][column] = ch;
            return boardCopy;
        })

        // return win value
        return checkWin(row, column, ch);
    }
    const swapPlayers = () => {
        if (!gameOver) {
            setCurrPlayer(prev => {
                setOppPlayer(prev);
                return oppPlayer;
            })
        }    
    }
    const handleClick = (e) => {
        console.log('board clicked');

        //get col and row of the player's click
        const column = e.target.getAttribute('x');
        let row = board.findIndex((rowArr, index) => {
            //finding the occupied spot in the column || bottom of the column
            return (rowArr[column] !== '' ||
                (index === board.length - 1));
        })
        //if _row_ is not the last row, move one up (-=1)
        if (row !== board.length - 1) row -= 1;
        //if slot at (row,col) is occupied, move one up
        if (board[row][column] !== '') row -= 1;

        //update board based on move
        setGameOver(updateBoard(row, column, currPlayer));

        /* swap players */
        swapPlayers();
    };

    //BACKEND
    const getModelMove = async (currentBoard) => {
        try {
            const response = await axios.post("http://localhost:5000/get-model-move", {
                board: currentBoard,  // 6x7 array of '', 'X', 'O'
                difficulty: difficulty    // Or dynamic difficulty
            });

            const { row, column } = response.data;
            console.log(`Model played row=${row} col=${column}`);

            setGameOver(updateBoard(row, column, currPlayer));

            // Swap turns after the model's move
            swapPlayers();

        } catch (error) { //randomly play a move
            console.error("Error getting model move:", error);
            
            const column = Math.floor(Math.random() * 7);
            //find row for token given _column_
            let row = board.findIndex((rowArr, index) => {
                return (rowArr[column] !== '' ||
                    (index === board.length - 1));
            })
            if (row !== board.length - 1) row -= 1;
            if (board[row][column] !== '') row -= 1;

            setGameOver(updateBoard(row, column, currPlayer));
            swapPlayers();
        }
    };


    return (
        <>
            {/* Difficulty selector */}
            {!gameOver && (
                <div>
                    <label>Select Difficulty: </label>
                    <select value={difficulty} onChange={handleDifficultyClick}>
                        <option value="easy">Easy</option>
                        <option value="medium">Medium</option>
                        <option value="hard">Hard</option>

                    </select>
                </div>
            )}
            {gameOver && (
                <h1>Game Over! {oppPlayer === 'X' ? 'Red' : 'Model'} Wins!</h1>
            )}
            {!gameOver && (
                <h2 id='playerDisplay'>{currPlayer === 'X' ? 'Your' : 'Computer'} Move</h2>
            )}
            <div id='board'
                onClick={gameOver ? null : handleClick}
            >
                {board.map((row, i) => {
                    //small error, keys needed for each slot
                    return row.map((ch, j) => <Slot key={`${i}-${j}`}ch={ch} y={i} x={j} />);
                })}
            </div>
        </>
    )
};

export default Board