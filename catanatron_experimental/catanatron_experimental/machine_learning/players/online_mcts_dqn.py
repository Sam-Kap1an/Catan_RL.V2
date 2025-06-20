from collections import deque
from catanatron_experimental.data_logger import DataLogger
import os
import random
import time
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf

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
MODEL_NAME = "online-mcts-dqn-3.0"
MODEL_PATH = str(Path("data/models/", MODEL_NAME))
MODEL_SINGLETON = None
DATA_LOGGER = DataLogger(DATA_PATH)



import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_action = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc_action(x)
        return q_values

MODEL_SINGLETON = None

def get_model():
    global MODEL_SINGLETON
    if MODEL_SINGLETON is None:
        model = QNetwork(input_dim=NUM_FEATURES, action_dim=1) 
        MODEL_SINGLETON = model
    return MODEL_SINGLETON


# class OnlineMCTSDQNPlayer(Player):
#     def __init__(self, color):
#         super().__init__(color)
#         self.step = 0

#     def decide(self, game: Game, playable_actions):
#         """
#         For each move, will run N playouts, get statistics, and save into replay buffer.
#         Every M decisions, will:
#             - flush replay buffer to disk (for offline experiments)
#             - report progress on games thus far to TensorBoard (tf.summary module)
#             - update model by choosing L random samples from replay buffer
#                 and train model. do we need stability check? i think not.
#                 and override model path.
#         Decision V1 looks like, predict and choose the one that creates biggest
#             'distance' against enemies. Actually this is the same as maximizing wins.
#         Decision V2 looks the same as V1, but minimaxed some turns in the future.
#         """
#         if len(playable_actions) == 1:  # this avoids imbalance (if policy-learning)
#             return playable_actions[0]

#         start = time.time()

#         # Run MCTS playouts for each possible action, save results for training.
#         samples = []
#         scores = []
#         print(playable_actions)
#         for action in playable_actions:
#             print("Considering", action)
#             action_applied_game_copy = game.copy()
#             action_applied_game_copy.execute(action)
#             sample = create_sample_vector(action_applied_game_copy, self.color)
#             samples.append(sample)

#             if TRAIN:
#                 # Save snapshots from the perspective of each player (more training!)
#                 counter = run_playouts(action_applied_game_copy, NUM_PLAYOUTS)
#                 mcts_labels = {k: v / NUM_PLAYOUTS for k, v in counter.items()}
#                 DATA_LOGGER.consume(action_applied_game_copy, mcts_labels)

#                 scores.append(mcts_labels.get(self.color, 0))

#         # TODO: if M step, do all 4 things.
#         if TRAIN and self.step % FLUSH_EVERY == 0:
#             self.update_model_and_flush_samples()

#         # scores = get_model().call(tf.convert_to_tensor(samples))
#         best_idx = np.argmax(scores)
#         best_action = playable_actions[best_idx]

#         if TRAIN:
#             print("Decision took:", time.time() - start)
#         self.step += 1
#         return best_action

#     def update_model_and_flush_samples(self):
#         """Trains using NN, and saves to disk"""
#         global MIN_REPLAY_BUFFER_LENGTH, BATCH_SIZE, MODEL_PATH, OVERWRITE_MODEL

#         samples, board_tensors, labels = DATA_LOGGER.get_replay_buffer()
#         if len(samples) < MIN_REPLAY_BUFFER_LENGTH:
#             return

#         # print("Training...")
#         # model = get_model()
#         # model.fit(
#         #     tf.convert_to_tensor(samples),
#         #     tf.convert_to_tensor(labels),
#         #     batch_size=BATCH_SIZE,
#         #     verbose=0,
#         #     shuffle=True,
#         # )
#         # print("DONE training")
#         # if OVERWRITE_MODEL:
#         #     model.save(MODEL_PATH)

#         DATA_LOGGER.flush()


EPSILON = 0.1
GAMMA = 0.99  # Discount factor

class OnlineMCTSDQNPlayer_1(Player):
    def __init__(self, color):
        super().__init__(color)
        self.step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = deque(maxlen=5000)  # store (s, r, s') transitions

    def decide(self, game: Game, playable_actions):
        start = time.time()
        if len(playable_actions) == 1:
            print("only action")
            return playable_actions[0]

        samples = []
        next_states = []
        actions = []
        rewards = []

        for action in playable_actions:
            print("loop actions:", str(action))
            game_copy = game.copy()
            game_copy.execute(action)
            sample = create_sample_vector(game_copy, self.color)
            samples.append(sample)
            next_states.append(game_copy)
            actions.append(action)

            if TRAIN:
                # Reward is from MCTS estimate
                print("eval action:", str(action))
                counter = run_playouts(game_copy, NUM_PLAYOUTS)
                mcts_labels = {k: v / NUM_PLAYOUTS for k, v in counter.items()}
                reward = mcts_labels.get(self.color, 0)
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
        if TRAIN and random.random() < EPSILON:
            chosen_idx = random.randint(0, len(playable_actions) - 1)
            print("Exploration")
        else:
            chosen_idx = np.argmax(rewards)
            print("Exploitation")

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
        print(f"Eval took: {duration}.")
        print("selected action: ", str(actions[chosen_idx]))
        return actions[chosen_idx]
    

    def update_model_and_flush_samples(self):
        print("update_model_and_flush_samples")

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
                    q_val = self.model(ns_tensor).max(dim=1)[0].item()
                    next_qs.append(q_val)
            next_q_tensor = torch.tensor(next_qs, dtype=torch.float32).unsqueeze(1).to(self.device)

        targets = reward_tensor + GAMMA * next_q_tensor

        # Current predictions
        preds = self.model(state_tensor)

        loss = self.loss_fn(preds, targets)

        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if OVERWRITE_MODEL:
            torch.save(self.model.state_dict(), MODEL_PATH + ".pt")

        print(f"Trained DQN model with loss: {loss.item():.4f}")