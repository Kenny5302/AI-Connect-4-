# AI Connect 4 (React + Python/PyTorch)

## Overview
A full-stack **Connect 4** game with a React front end and a **PyTorch** backend that predicts the next best board state. The AI supports multiple difficulty levels (`easy`, `medium`, `hard`) via pre-trained models.

## Features
- **Interactive React UI** with a styled 7×6 board (`#board`, `.slot`) and smooth piece transitions.   
- **Neural network policy** that evaluates board states and proposes the next move; models saved as `.pth` files.   
- **Difficulty modes** that load different models at runtime (`easy_model.pth`, `medium_model.pth`, `hard_model.pth`).   
- **Benchmark script** to evaluate model accuracy, precision/recall, and F1 on a held-out dataset. 

## Tech Stack
- **Frontend:** React, JavaScript, CSS  
- **Backend / AI:** Python, PyTorch, TorchMetrics, NumPy  
- **Data:** `c4_game_database.csv` (board states + labels) 

