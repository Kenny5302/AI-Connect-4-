ECS171 Final Project - Deep Network for playing Connect-4

To train the model run the connect4deepnetwork.py file 
If you want to change the model open the connect4deepnetwork.py file and you can change the batch sizes for both training and testing,
how many epochs it is trained/tested for, and the number of layers/number of nodes in each layer.

To interact with the model run the connect4.py file which takes in a current board state in the form of an array and it will tell you the best move for the model to make and give you the updated board state. In case of more than one best move it randomly selects one, and in case of no best move it finds a move that will tie. In case of no tie moves it then randomly selects a move to take.
