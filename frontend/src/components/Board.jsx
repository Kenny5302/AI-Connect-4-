import Slot from "./Slot";
import {useState} from "react";
//BACKEND: import axios from "axios";  // Import axios for HTTP requests

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

    const checkWin = (row, column, ch) => {
        /*down 4*/
        try{
            if(board[row + 1][column] === ch) {
                if(board[row + 2][column] === ch) {
                    if(board[row + 3][column] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) {console.log(e)}
        
        /* side right 4 */
        try{
            if(board[row][column + 1] === ch) {
                if(board[row][column + 2] === ch) {
                    if(board[row][column + 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) {console.log(e)}

        /* side left 4 */
        try{
            if(board[row][column - 1] === ch) {
                if(board[row][column - 2] === ch) {
                    if(board[row][column - 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) {console.log(e)}

        /* diagonal down right 4 */
        try{
            if(board[row + 1][column + 1] === ch) {
                if(board[row + 2][column + 2] === ch) {
                    if(board[row + 3][column + 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) {console.log(e)}

        /* diagonal down left 4 */
        try{
            if(board[row + 1][column - 1] === ch) {
                if(board[row + 2][column - 2] === ch) {
                    if(board[row + 3][column - 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) {console.log(e)}

        /* diagonal upper left 4 */
        try{
            if(board[row-1][column - 1] === ch) {
                if(board[row-2][column - 2] === ch) {
                    if(board[row-3][column - 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) {console.log(e)}  

        /* diagonal upper right 4 */
        try{
            if(board[row-1][column + 1] === ch) {
                if(board[row-2][column + 2] === ch) {
                    if(board[row-3][column + 3] === ch) {
                        return true;
                    }
                }
            }
        } catch (e) {console.log(e)} 
    }

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

    const handleClick = (e) => {
        console.log('board clicked');

        //get col and row of the player's click
        const column = e.target.getAttribute('x');
        let row = board.findIndex((rowArr, index) => {
            //finding bottommost slot token can go into
            return (rowArr[column] !== '' || 
            (index === board.length - 1));
        })
        if (row !== board.length - 1) row -= 1;
        if (board[row][column] !== '') row -= 1;

        //update board based on move
        setGameOver(updateBoard(row, column, currPlayer));

        /* swap players */
        if(!gameOver){
            const currPlayerCopy = currPlayer;
            setCurrPlayer(oppPlayer);
            setOppPlayer(currPlayerCopy);
            
            /*BACKEND: MUST WAIT FOR REQUEST FOR MODEL'S MOVE */
            getModelMove(board)
        }

    };

    //BACKEND    
    const getModelMove = async (currentBoard) => {
        try {
            
            // BACKEND: SAMPLE BACKEND REQUEST CODE
            const response = (1, 1); //Delete 
            /*
            const response = await axios.post("http://your-backend-url/get-model-move", {
                board: currentBoard, 
                //BACKEND: board REPRESENTED AS 6X7 ARRAY OF '' OR 'X' OR 'O'. MODEL PLAYS 'O'
            });
            */

            //BACKEND: THE MOVE WE RECIEVE FROM MODEL
            const { row, column } = response.data;

            setGameOver(updateBoard(row, column, currPlayer));

            // Swap turns after the model's move
            if (!gameOver) {
                const currPlayerCopy = currPlayer;
                setCurrPlayer(oppPlayer);
                setOppPlayer(currPlayerCopy);
            }

        } catch (e) {
            console.log("Error getting model move", e);
        }
    };

    return (
        <>
            {gameOver && (
                <h1>Game Over! {oppPlayer === 'X' ? 'Red' : 'Model'} Wins!</h1>
            )}
            <h2 id='playerDisplay'>{currPlayer === 'X' ? 'Your' : 'Computer'} Move</h2>
            <div id='board'
                onClick={gameOver ? null : handleClick}
            >
                {board.map((row, i) => {
                    return row.map((ch, j) => <Slot ch={ch} y={i} x={j} />);
                })}
            </div>
        </>
    )
};

export default Board