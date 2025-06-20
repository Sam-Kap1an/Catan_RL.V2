from collections import deque
from catanatron_experimental.data_logger import DataLogger
import os
import random
import time
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer

from tensorflow.keras.optimizers import Adam

from catanatron.game import Game
from catanatron.models.player import Player
from catanatron_experimental.machine_learning.players.playouts import run_playouts
from catanatron_gym.features import (
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_gym.board_tensor_features import (
    WIDTH,
    HEIGHT,
    CHANNELS,
    create_board_tensor,
)

# ===== CONFIGURATION
NUM_FEATURES = len(get_feature_ordering())
NUM_PLAYOUTS = 100
MIN_REPLAY_BUFFER_LENGTH = 100
BATCH_SIZE = 64
FLUSH_EVERY = 1  # decisions. what takes a while is to generate samples via MCTS
TRAIN = True
OVERWRITE_MODEL = True
DATA_PATH = "data/mcts-playouts-validation"
NORMALIZATION_MEAN_PATH = Path(DATA_PATH, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(DATA_PATH, "variance.npy")

# ===== PLAYER STATE (here to allow pickle-serialization of player)
MODEL_NAME = "MM-dqn-1.0"
MODEL_PATH = str(Path("data/models/", MODEL_NAME))
MODEL_SINGLETON = None
DATA_LOGGER = DataLogger(DATA_PATH)



import torch
import torch.nn as nn
import torch.nn.functional as F



class MMQNetwork(nn.Module): 
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.board_shape = (CHANNELS, HEIGHT, WIDTH)
        self.board_flat_size = np.prod(self.board_shape)
 # CNN branch
        self.cnn = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        dummy_input = torch.zeros(1, *self.board_shape)
        cnn_output_dim = self.cnn(dummy_input).view(1, -1).shape[1]


        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        board_tensor = x[:, :self.board_flat_size].reshape(-1, *self.board_shape)
        feature_vector = x[:, self.board_flat_size:]

        cnn_out = self.cnn(board_tensor).view(board_tensor.size(0), -1)
        mlp_out = self.mlp(feature_vector)

        combined = torch.cat([cnn_out, mlp_out], dim=1)
        return self.fc(combined)
    
    
def create_combined_input(game, color, features=None):
    vector = create_sample_vector(game, color, features)
    board = create_board_tensor(game, color, channels_first=True).flatten()
    vector = create_sample_vector(game, color, features)
    return np.concatenate([board, vector])



MODEL_SINGLETON = None


def get_model_and_epsilon(device):
    global MODEL_SINGLETON
    if MODEL_SINGLETON is None:
        model = QNetwork(input_dim=NUM_FEATURES, action_dim=1).to(device)
        epsilon = 1.0  # default starting value

        if os.path.exists(MODEL_PATH + ".pt"):
            checkpoint = torch.load(MODEL_PATH + ".pt", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            epsilon = checkpoint.get('epsilon', 1.0)
            print(f"Loaded model with epsilon = {epsilon:.4f}")
        else:
            print("No saved model found. Starting fresh.")

        MODEL_SINGLETON = (model, epsilon)

    return MODEL_SINGLETON


EPSILON = 0.1
GAMMA = 0.99  # Discount factor



def evaluate_with_alphabeta(game, color, depth=2):
    player = AlphaBetaPlayer(color=color, depth=depth, prunning=True)
    actions = game.state.playable_actions
    if not actions:
        return 0.0  # If no actions, no reward
    best_action, value = player.alphabeta(game, depth, float("-inf"), float("inf"), time.time() + 2, None)
    return value


class MM_DQNPlayer_1(Player):
    def __init__(self, color):
        super().__init__(color)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.epsilon = get_model_and_epsilon(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=5000)  # store (s, r, s') transitions

    def decide(self, game: Game, playable_actions):
        start = time.time()
        if len(playable_actions) == 1:
            # print("only action")
            return playable_actions[0]

        samples = []
        next_states = []
        actions = []
        rewards = []

        for action in playable_actions:
            #print("loop actions:", str(action))
            game_copy = game.copy()
            game_copy.execute(action)
            sample = create_sample_vector(game_copy, self.color)
            samples.append(sample)
            next_states.append(game_copy)
            actions.append(action)

            if TRAIN:
                # Reward is from MCTS estimate
                #print("eval action:", str(action))
                reward = evaluate_with_alphabeta(game_copy, self.color, depth=2)
                reward = np.clip(reward, -10, 10)
                rewards.append(reward)
                # Store (sample, reward, next_sample) for Bellman update
                self.replay_buffer.append((sample, reward, None))  # next_sample will be filled below
            else:
                # Predict using current Q-network
                tensor_input = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_value = self.model(tensor_input).item()
                rewards.append(q_value)

        # Epsilon-greedy action selection
        if TRAIN and random.random() < self.epsilon:
            chosen_idx = random.randint(0, len(playable_actions) - 1)
        else:
            chosen_idx = np.argmax(rewards)

        # Decay epsilon over time
        if TRAIN:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


        if TRAIN:
            # Fill in next_sample for chosen action
            chosen_game = next_states[chosen_idx]
            next_sample = create_sample_vector(chosen_game, self.color)
            # Update last appended (sample, reward, None) with next_sample
            self.replay_buffer[-1] = (samples[chosen_idx], rewards[chosen_idx], next_sample)

        if TRAIN and self.step % FLUSH_EVERY == 0:
            self.update_model_and_flush_samples()

        self.step += 1
        duration = time.time() - start
        #print(f"Eval took: {duration}.")
        # print("selected action: ", str(actions[chosen_idx]))
        return actions[chosen_idx]
    

    def update_model_and_flush_samples(self):
        #print("update_model_and_flush_samples")

        if len(self.replay_buffer) < MIN_REPLAY_BUFFER_LENGTH:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, rewards, next_states = zip(*batch)

        state_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # For terminal states, next_state is None, so handle that
            next_qs = []
            for ns in next_states:
                if ns is None:
                    next_qs.append(0.0)
                else:
                    ns_tensor = torch.tensor(ns, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_val = self.model(ns_tensor).item()
                    next_qs.append(q_val)
            next_q_tensor = torch.tensor(next_qs, dtype=torch.float32).unsqueeze(1).to(self.device)

        targets = reward_tensor + GAMMA * next_q_tensor

        # Current predictions
        preds = self.model(state_tensor)

        loss = self.loss_fn(preds, targets)

        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        if OVERWRITE_MODEL:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'epsilon': self.epsilon
            }, MODEL_PATH + ".pt")

       # print(f"Trained DQN model with loss: {loss.item():.4f}")