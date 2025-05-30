# ====== Imports and Config ======
from collections import deque
from catanatron_experimental.data_logger import DataLogger
import os
import random
import time
from pathlib import Path
import copy

import pandas as pd
import numpy as np
import tensorflow as tf
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from tensorflow.keras.optimizers import Adam
from catanatron.game import Game
from catanatron.models.player import Player
from catanatron_experimental.machine_learning.players.playouts import run_playouts
from catanatron_gym.features import create_sample_vector, get_feature_ordering
from catanatron_gym.board_tensor_features import WIDTH, HEIGHT, CHANNELS, create_board_tensor

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Configuration =====
NUM_FEATURES = len(get_feature_ordering())
NUM_PLAYOUTS = 100
MIN_REPLAY_BUFFER_LENGTH = 100
BATCH_SIZE = 256
FLUSH_EVERY = 1
TRAIN = True
OVERWRITE_MODEL = True
DATA_PATH = "data/mcts-playouts-validation"
NORMALIZATION_MEAN_PATH = Path(DATA_PATH, "mean.npy")
NORMALIZATION_VARIANCE_PATH = Path(DATA_PATH, "variance.npy")
MODEL_NAME = "AB-dqn-4.0"
MODEL_PATH = str(Path("data/models/", MODEL_NAME))
MODEL_SINGLETON = None
DATA_LOGGER = DataLogger(DATA_PATH)
GAMMA = 0.99

# ===== Neural Network Definition =====
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch normalization for stability
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_action = nn.Linear(hidden_dim, action_dim)
        # Initialize final layer to small values to avoid early explosion
        nn.init.uniform_(self.fc_action.weight, -0.01, 0.01)
        nn.init.zeros_(self.fc_action.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        q_values = torch.tanh(self.fc_action(x)) * 10  # Bound output to [-10, 10]
        return q_values

# ===== Model Loader with Epsilon =====
def get_model_and_epsilon(device):
    global MODEL_SINGLETON
    if MODEL_SINGLETON is None:
        model = QNetwork(input_dim=NUM_FEATURES, action_dim=1).to(device)
        epsilon = 1.0
        if os.path.exists(MODEL_PATH + ".pt"):
            checkpoint = torch.load(MODEL_PATH + ".pt", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            epsilon = checkpoint.get('epsilon', 1.0)
            print(f"Loaded model with epsilon = {epsilon:.4f}")
        else:
            print("No saved model found. Starting fresh.")
        MODEL_SINGLETON = (model, epsilon)
    return MODEL_SINGLETON

# ===== AlphaBeta Evaluation Wrapper =====
def evaluate_with_alphabeta(game, color, depth=2):
    player = AlphaBetaPlayer(color=color, depth=depth, prunning=True)
    actions = game.state.playable_actions
    if not actions:
        return 0.0
    best_action, value = player.alphabeta(game, depth, float("-inf"), float("inf"), time.time() + 2, None)
    return value

# ===== DQN Player Implementation =====
class AB_DQNPlayer_1(Player):
    def __init__(self, color):
        super().__init__(color)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Epsilon-greedy exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.step = 0
        self.update_target_steps = 100  # How often to sync target network

        # Load or initialize model and target
        self.model, self.epsilon = get_model_and_epsilon(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.eval()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.SmoothL1Loss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)

        # Load or initialize replay buffer
        if os.path.exists("replay_buffer.pkl"):
            with open("replay_buffer.pkl", "rb") as f:
                self.replay_buffer = pickle.load(f)
            print(f"Loaded replay buffer with {len(self.replay_buffer)} transitions")
        else:
            self.replay_buffer = deque(maxlen=5000)

    def decide(self, game: Game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        samples, next_states, actions, rewards = [], [], [], []

        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)
            sample = create_sample_vector(game_copy, self.color)
            samples.append(sample)
            next_states.append(game_copy)
            actions.append(action)

            if TRAIN:
                reward = np.tanh(evaluate_with_alphabeta(game_copy, self.color, depth=2) / 10.0) * 10
                rewards.append(reward)
                self.replay_buffer.append((sample, reward, None))
            else:
                tensor_input = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_value = self.model(tensor_input).item()
                rewards.append(q_value)


        # ===== Hybrid Exploration Strategy =====
        e_total = self.epsilon
        e_random = e_total * 0.5       # 50% of e goes to random
        e_alpha = e_total * 0.5        # 50% of e goes to AlphaBeta
        r = random.random()

        
        # Epsilon-greedy action selection
        if TRAIN:
            if r < e_random:
                chosen_idx = random.randint(0, len(playable_actions) - 1)
            elif r < e_random + e_alpha:
                # Use AlphaBeta to pick the best action
                best_action, _ = AlphaBetaPlayer(self.color, depth=2, prunning=True).alphabeta(
                    game, 2, float("-inf"), float("inf"), time.time() + 1.5, None)
                chosen_idx = actions.index(best_action) if best_action in actions else random.randint(0, len(actions)-1)
            else:
                chosen_idx = np.argmax(rewards)
        else:
            chosen_idx = np.argmax(rewards)

        # Epsilon decay
        if TRAIN:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            chosen_game = next_states[chosen_idx]
            next_sample = create_sample_vector(chosen_game, self.color)
            self.replay_buffer[-1] = (samples[chosen_idx], rewards[chosen_idx], next_sample)

        if TRAIN and self.step % FLUSH_EVERY == 0:
            self.update_model_and_flush_samples()

        self.step += 1
        return actions[chosen_idx]

    def update_model_and_flush_samples(self):
        if len(self.replay_buffer) < MIN_REPLAY_BUFFER_LENGTH:
            return

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, rewards, next_states = zip(*batch)

        state_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_qs = []
            for ns in next_states:
                if ns is None:
                    next_qs.append(0.0)
                else:
                    ns_tensor = torch.tensor(ns, dtype=torch.float32).unsqueeze(0).to(self.device)
                    q_val = self.target_model(ns_tensor).item()
                    q_val = np.clip(q_val, -100, 100)
                    next_qs.append(q_val)
            next_q_tensor = torch.tensor(next_qs, dtype=torch.float32).unsqueeze(1).to(self.device)

        targets = reward_tensor + GAMMA * next_q_tensor
        
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
            with open("replay_buffer.pkl", "wb") as f:
                pickle.dump(self.replay_buffer, f)

        if self.step % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())

 
        self.lr_scheduler.step()
     

