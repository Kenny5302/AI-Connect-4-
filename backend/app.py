import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# from connect4deepnetwork import ConnectFourCNN
from flask import Flask, request
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app)

move_request = api.model(
    "MoveRequest",
    {
        "board": fields.List(fields.List(fields.String)),  # 6x7 board
        "difficulty": fields.String(required=True, enum=["easy", "medium", "hard"]),
    },
)


def convert_to_board_state(cols):
    return [cell for col in cols for cell in col]


# class ConnectFourCNN(nn.Module):
#     def __init__(self):
#         super(ConnectFourCNN, self).__init__()
#         # self.fc1 = nn.Linear(42, 300)
#         # self.fc2 = nn.Linear(300, 700)
#         # self.fc3 = nn.Linear(700, 300)
#         # self.fc4 = nn.Linear(300, 3)
#         self.fc1 = nn.Linear(42, 150)
#         self.fc2 = nn.Linear(150, 150)
#         self.fc3 = nn.Linear(150, 3)

#     def forward(self, x):
#         # x = self.fc1(x)
#         # x = F.relu(x)

#         # x = self.fc2(x)
#         # x = F.relu(x)

#         # x = self.fc3(x)
#         # x = F.relu(x)

#         # x = self.fc4(x)
#         # x = F.log_softmax(x, dim=-1)
#         # return x
#         x = self.fc1(x)
#         x = F.relu(x)

#         x = self.fc2(x)
#         x = F.relu(x)

#         x = self.fc3(x)
#         # print(x.shape)
#         x = F.log_softmax(x, dim=-1)
#         return x


class ConnectFourCNN(nn.Module):
    def __init__(self):
        super(ConnectFourCNN, self).__init__()
        # Convolutional layers to capture spatial patterns
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(128 * 6 * 7, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        # Reshape input to 2D board format
        x = x.view(-1, 1, 6, 7)

        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten for dense layers
        x = x.view(-1, 128 * 6 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)
        # x = self.fc1(x)
        # x = F.relu(x)


def find_best_move_1(board_state, difficulty):
    model_path = ""

    if difficulty == "hard":
        model_path = "hard_model.pth"
    elif difficulty == "medium":
        model_path = "medium_model.pth"
    else:
        model_path = "easy_model.pth"

    model = torch.load(model_path)

    model.eval()

    with torch.no_grad():
        col1 = []
        col2 = []
        col3 = []
        col4 = []
        col5 = []
        col6 = []
        col7 = []
        i = 1
        for entry in board_state:
            if i == 1:
                col1.append(entry)
                i += 1
            elif i == 2:
                col2.append(entry)
                i += 1
            elif i == 3:
                col3.append(entry)
                i += 1
            elif i == 4:
                col4.append(entry)
                i += 1
            elif i == 5:
                col5.append(entry)
                i += 1
            elif i == 6:
                col6.append(entry)
                i += 1
            elif i == 7:
                col7.append(entry)
                i = 1
        column_array = [col1, col2, col3, col4, col5, col6, col7]
        index = -1
        attemped_board_states = []
        winning_board_states = []
        tie_board_states = []
        board_state_to_return = []
        for col in column_array:
            for i, entry in enumerate(col):
                if entry == 0:
                    if col[i + 1] != 0:
                        index = i
                        break
            if index != -1:
                col[index] = 2.0
                new_board_state = convert_to_board_state(
                    col1, col2, col3, col4, col5, col6, col7
                )
                attemped_board_states.append(new_board_state)
                new_board_state_tensor = torch.tensor(new_board_state)
                winner = model(new_board_state_tensor)
                winner = winner.argmax(1)
                winner = int(winner.numpy())
                if winner == 2:
                    winning_board_states.append(new_board_state)
                elif winner == 0:
                    tie_board_states.append(new_board_state)
                col[index] = 0.0
                index = -1
        if not winning_board_states:
            if not tie_board_states:
                random_state = random.randrange(1, 8)
                board_state_to_return = attemped_board_states[random_state]
            else:
                num_tie_boards = len(tie_board_states)
                if num_tie_boards == 1:
                    board_state_to_return = tie_board_states[0]
                else:
                    random_state = random.randrange(1, num_tie_boards)
                    board_state_to_return = tie_board_states[random_state]
        else:
            num_winning_boards = len(winning_board_states)
            if num_winning_boards == 1:
                board_state_to_return = winning_board_states[0]
            else:
                random_state = random.randrange(1, num_winning_boards)
                board_state_to_return = winning_board_states[random_state]

        return board_state_to_return


def find_best_move(board_state, difficulty):
    model_path = f"{difficulty}_model.pth"
    model = torch.load("./test_model.pth", weights_only=False)
    model.eval()

    cols = [board_state[i::7] for i in range(7)]
    attempted = []

    for col_idx in range(7):
        col = cols[col_idx].copy()
        try:
            row = next(
                i for i, v in enumerate(col) if v == 0.0 and (i == 5 or col[i + 1] != 0)
            )
        except StopIteration:
            continue

        new_col = col.copy()
        new_col[row] = 2.0
        new_cols = cols.copy()
        new_cols[col_idx] = new_col
        new_state = convert_to_board_state(zip(*new_cols))

        with torch.no_grad():
            output = model(torch.tensor(new_state, dtype=torch.float32))
            print(col_idx, output)
            pred = output.argmax().item()

        attempted.append((new_state, col_idx, pred))

    winning = [a for a in attempted if a[2] == 2]
    ties = [a for a in attempted if a[2] == 0]
    others = [a for a in attempted if a[2] == 1]

    if winning:
        chosen = random.choice(winning)
        print(f"choosing winning move {chosen[1]} from {[x[1] for x in winning]}")
    elif ties:
        chosen = random.choice(ties)
        print(f"choosing tie move {chosen[1]} from {[x[1] for x in ties]}")
    elif others:
        chosen = random.choice(others)
        print(f"choosing losing move {chosen[1]} from {[x[1] for x in others]}")
    else:
        print("no valid moves")
        return None

    return chosen[0], chosen[1]


@api.route("/get-model-move")
class ModelMove(Resource):
    @api.expect(move_request)
    def post(self):
        data = request.json
        frontend_board = data["board"]
        difficulty = data["difficulty"]

        model_input = []
        for row in frontend_board:
            for cell in row:
                model_input.append(1.0 if cell == "X" else 2.0 if cell == "O" else 0.0)

        result = find_best_move(model_input, difficulty)
        if not result:
            return {"error": "No valid moves"}, 400

        new_state, column = result
        diff = [
            i
            for i, (orig, new) in enumerate(zip(model_input, new_state))
            if orig != new
        ]
        row = diff[0] // 7 if diff else 0

        return {"row": row, "column": column}


if __name__ == "__main__":
    app.run(debug=True)
