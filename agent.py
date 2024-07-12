import torch
import random
import numpy as np
from collections import deque

from snake_game import SnakeGame, Direction, Point, BLOCK_SIZE
from model import LinearQNet, QTrainer
from utils import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.0001
GAMMA = 0.9


class Agent:
    def __init__(self) -> None:
        self.epsilon = 0
        self.num_games = 0
        self.gamma = GAMMA
        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game: SnakeGame):
        # 11 Values (danger straight, danger right, danger left, direction lrud, food lrud)
        head = game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        danger_straight = (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(
            point_l)) or (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d))
        danger_right = (dir_r and game.is_collision(point_d)) or (dir_l and game.is_collision(
            point_u)) or (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l))
        danger_left = (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(
            point_d)) or (dir_u and game.is_collision(point_l)) or (dir_d and game.is_collision(point_r))

        state = [
            danger_straight,
            danger_right,
            danger_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            game.food.x < head.x,
            game.food.x > head.x,
            game.food.y < head.y,
            game.food.y > head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        num_samples = min(BATCH_SIZE, len(self.memory))
        samples = random.sample(self.memory, num_samples)

        states, actions, rewards, next_states, game_overs = zip(*samples)
        self.trainer.train_step(states, actions, rewards,
                                next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        self.epsilon = 80 - self.num_games
        action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state = torch.tensor(state, dtype=torch.float)
            move = self.model(state)
            move = torch.argmax(move).item()
            action[move] = 1

        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record_score = 0

    agent = Agent()
    game = SnakeGame()

    while True:
        state = agent.get_state(game)

        action = agent.get_action(state)
        reward, game_over, score = game.play_step(action)
        new_state = agent.get_state(game)

        agent.train_short_memory(state, action, reward, new_state, game_over)
        agent.remember(state, action, reward, new_state, game_over)

        if game_over:
            agent.train_long_memory()

            game.reset()
            agent.num_games += 1

            # Plot Results
            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.num_games)
            plot(plot_scores, plot_mean_scores)

            if score > record_score:
                record_score = score

            print(
                f"GAME {agent.num_games} || Score: {score} , Record: {record_score}")
            # Save Model


if __name__ == "__main__":
    train()
