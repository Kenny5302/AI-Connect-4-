import React, { useState } from 'react';
import './board.css';

function connectBoard() {
  /*const [inputData, setInputData] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(inputData);
    // Send inputData to the backend API here
  };

  return (
    <div>
      <h2>Enter Data for Prediction</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputData}
          onChange={(e) => setInputData(e.target.value)}
          placeholder="Enter some data"
        />
        <button type="submit">Get Prediction</button>
      </form>
    </div>
  );*/

  return (
    <div className="display">
         <div className="board">Board</div>
    </div>
  );
}

export default connectBoard;