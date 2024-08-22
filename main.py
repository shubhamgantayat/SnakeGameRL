from snakegame_rl import SnakeGame
from tqdm import tqdm
import numpy as np


game = SnakeGame(n=10)
reward_matrix = np.zeros((game.board.shape[0] * game.board.shape[1], 4, game.board.shape[0] * game.board.shape[1]))
scores = []
rewards = []
for _ in tqdm(range(1000)):
    game = SnakeGame(n=10)
    while True:
        check = game.train(reward_matrix)
        if check == -1:
            break
        else:
            rewards.append(check)
    scores.append(game.snake_len)

print("STARTING GAME")
game = SnakeGame(10)
game.start(reward_matrix)
print(game.snake_len)