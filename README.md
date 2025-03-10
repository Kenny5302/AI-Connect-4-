ECS171 Final Project - Deep Network for playing Connect-4

To train the model run the connect4deepnetwork.py file 
If you want to change the model open the connect4deepnetwork.py file and you can change the batch sizes for both training and testing,
how many epochs it is trained/tested for, and the number of layers/number of nodes in each layer.

To interact with the model run the connect4.py file which takes in a current board state in the form of an array and will tell you the winner of the game
(Going to make it so it takes in a board state and then runs through all possible current moves and then selects which move is the best based off of which 
move will make it win and then returns that move/updated board state)
