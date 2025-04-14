import os
import numpy as np
import imageio
import pickle
import torch
from tqdm import tqdm
import sys
sys.path.append("flappy-bird-gym")
import flappy_bird_gym
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple
from train_flappy_bc import DiscretePolicy  # your trained policy class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/scratch/gpfs/cz5047/flappy_color_data"

# Load the trained model
policy = DiscretePolicy(input_dim=6, n_actions=2)
policy.load_state_dict(torch.load("bc_trained_policy.pth", map_location=device, weights_only=True))
policy.to(device)
policy.eval()

# Wrapper for inference
class TrainedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.device = next(policy.parameters()).device
    def __call__(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.policy(state_tensor)

# Rollout + render function
def rollout_and_render(env, policy):
    frames = []
    actions = []

    state = env.reset()
    done = False

    while not done:
        frame = env.render(mode="rgb_array")
        frame = np.transpose(frame, (1, 0, 2))
        frames.append(frame)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(policy.device)
        action_dist = policy(state_tensor)
        action = action_dist.sample().item()
        actions.append(torch.tensor(int(action)))

        state, _, done, _ = env.step(action)

    return frames, actions

# Config
bird_colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "white", "brown"]
bird_colors_subset = bird_colors[8:]  # Use only the first 5 colors for testing
n_rollouts = 100

# Main loop with tqdm
for color in bird_colors_subset:
    print(f"\n🎮 Generating rollouts for bird color: {color}")
    color_dir = BASE_DIR
    os.makedirs(color_dir, exist_ok=True)

    for i in tqdm(range(1, n_rollouts + 1), desc=f"{color}", ncols=80):
        env = FlappyBirdEnvSimple(bird_color=color, background="day")
        frames, actions = rollout_and_render(env, TrainedPolicy(policy))
        env.close()

        video_path = os.path.join(color_dir, f"{color}_{i}.mp4")
        action_path = os.path.join(color_dir, f"{color}_{i}.pkl")

        imageio.mimsave(video_path, frames, fps=30)
        with open(action_path, "wb") as f:
            pickle.dump(actions, f)
