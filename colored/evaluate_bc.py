import os
import numpy as np
import imageio
import pickle
import torch

import sys
sys.path.append("flappy-bird-gym")
import flappy_bird_gym
from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple

# Your trained PyTorch policy (from previous code)
# Make sure it's on the right device and in `.eval()` mode
from train_flappy_bc import DiscretePolicy  # or wherever your class is defined

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rebuild the model architecture
policy = DiscretePolicy(input_dim=6, n_actions=2)  # 🔁 Adjust if input_dim is not 2!
policy.load_state_dict(torch.load("bc_trained_policy.pth", map_location=device, weights_only=True))
policy.to(device)
policy.eval()

from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

rc('animation', html='jshtml')

def animate_images(img_lst):
  fig, ax = plt.subplots()
  ax.set_axis_off()
  ims = []
  for i, image in enumerate(img_lst):
    im = ax.imshow(image.swapaxes(0, 1), animated=True)
    if i == 0:
      ax.imshow(image.swapaxes(0, 1))
    ims.append([im])
  ani = animation.ArtistAnimation(fig, ims, interval=34)
  return ani

from typing import List

def rollout_and_render(env, policy) -> List[np.ndarray]:
  """Returns a rendering of a single rollout under the provided policy.

  Args:
    env: OpenAI gym environment following old API.
    policy: Callable representing a policy.

  Returns:
    A list of RGB images (as numpy arrays) from a single rollout.
  """
  ### YOUR CODE HERE!

  img_lst = []  # List to store frames
  frames = []
  actions = []
  state = env.reset()
  done = False

  while not done:
      # Capture frame before taking action
      img_lst.append(env.render(mode="rgb_array"))
      frame = env.render(mode="rgb_array")
      frame = np.transpose(frame, (1, 0, 2))
      frames.append(frame)

      # Get action from policy
      state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
      action_dist = policy(state_tensor)
      action = action_dist.sample().item()
      actions.append(torch.tensor(int(action)))

      # Step in the environment
      state, _, done, _ = env.step(action)

  return img_lst, frames, actions

# env = flappy_bird_gym.make("FlappyBird-v0")
env = FlappyBirdEnvSimple(bird_color="green", background="day")


# define a callable policy to evaluate the training
class TrainedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.device = next(policy.parameters()).device 
    def __call__(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.policy(state_tensor)


img_lst, frames, actions = rollout_and_render(env, TrainedPolicy(policy))    # Assign list of images here

# ### DO NOT MODIFY ANYTHING BELOW THIS POINT
# ani = animate_images(img_lst)
# FFwriter = animation.FFMpegWriter(fps=30)
# ani.save('animation.mp4', writer=FFwriter)

# Save video
# print(frames)
video_path = "videos/flappy_bc2_model.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"🎥 Saved video to: {video_path}")

# Save actions
action_path = "videos/flappy_bc2_actions.pkl"
with open(action_path, "wb") as f:
    pickle.dump(actions, f)
print(f"✅ Saved actions to: {action_path}")



# # Set up dummy drivers (no sound or screen needed)
# os.environ["SDL_AUDIODRIVER"] = "dummy"
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.makedirs("videos", exist_ok=True)

# # Create environments
# env_policy = flappy_bird_gym.make("FlappyBird-v0")
# env_render = FlappyBirdEnvSimple(bird_color="black", background="day")

# obs_policy = env_policy.reset()
# env_render.reset()

# frames = []
# actions = []

# last_frame = None
# unchanged_count = 0
# unchanged_threshold = 5
# step = 0

# while True:
#     # Convert observation to tensor and run through your policy
#     obs_tensor = torch.tensor(obs_policy, dtype=torch.float32).unsqueeze(0).to(device)
#     with torch.no_grad():
#         dist = policy(obs_tensor)
#         action = dist.sample().item()

#     actions.append(torch.tensor(int(action)))

#     # Step environments
#     obs_policy, reward, done, info = env_policy.step(action)
#     obs_policy = obs_policy.astype(np.float32)
#     env_render.step(action)
#     frame = env_render.render(mode="rgb_array")
#     frame = np.transpose(frame, (1, 0, 2))

#     # Detect frozen frame
#     if last_frame is not None and np.array_equal(frame, last_frame):
#         unchanged_count += 1
#     else:
#         unchanged_count = 0

#     last_frame = frame.copy()
#     frames.append(frame)

#     if unchanged_count >= unchanged_threshold:
#         print(f"🧊 Frame frozen at step {step} — exiting.")
#         break

#     if done:
#         print(f"🛑 Env done=True at step {step}")
#         break

#     step += 1

# # Close envs
# env_policy.close()
# env_render.close()

# # Save video
# video_path = "videos/flappy_bc_model.mp4"
# imageio.mimsave(video_path, frames, fps=30)
# print(f"🎥 Saved video to: {video_path}")

# # Save actions
# action_path = "videos/flappy_bc_actions.pkl"
# with open(action_path, "wb") as f:
#     pickle.dump(actions, f)
# print(f"✅ Saved actions to: {action_path}")

################################################
# import os
# import torch
# import numpy as np
# import imageio
# import pickle
# import sys
# sys.path.append("flappy-bird-gym")
# import flappy_bird_gym

# from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
# from flappy_bird_gym.envs.flappy_bird_env_simple import FlappyBirdEnvSimple
# from train_flappy_bc import DiscretePolicy  # or wherever your model class lives

# # === Setup ===
# os.environ["SDL_AUDIODRIVER"] = "dummy"
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# os.makedirs("videos", exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # === Load Policy ===
# input_dim = 6        # or 2, depending on what your model was trained on
# n_actions = 2        # flappy bird is binary: [0, 1]

# policy = DiscretePolicy(input_dim=input_dim, n_actions=n_actions)
# policy.load_state_dict(torch.load("bc_trained_policy.pth", map_location=device))
# policy.to(device)
# policy.eval()

# # === Environment ===
# env_policy = flappy_bird_gym.make("FlappyBird-v0")
# env_render = FlappyBirdEnvSimple(bird_color="black", background="day")

# obs = env_policy.reset()
# ceck = env_render.reset()

# # print("Obs shape:", ceck.shape)
# # print("Obs shape:", obs.shape)


# frames = []
# actions = []

# done = False
# step = 0
# unchanged_count = 0
# unchanged_threshold = 100
# last_frame = None

# # === Rollout loop ===
# while not done:
#     # === Match rollout_and_render style ===
#     obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
#     with torch.no_grad():
#         dist = policy(obs_tensor)
#         action = dist.sample().item()
    
#     actions.append(torch.tensor(int(action)))

#     # Step both envs
#     obs, reward, done, _ = env_policy.step(action)
#     frame = env_render.render(mode="rgb_array")
#     frame = np.transpose(frame, (1, 0, 2))

#     # Check for frozen frames
#     if last_frame is not None and np.array_equal(frame, last_frame):
#         unchanged_count += 1
#     else:
#         unchanged_count = 0
#     last_frame = frame.copy()

#     frames.append(frame)

#     if unchanged_count >= unchanged_threshold:
#         print(f"🧊 Frame frozen at step {step} — exiting.")
#         break

#     if done:
#         print(f"🛑 Env done=True at step {step}")
#         break

#     step += 1

# env_policy.close()
# env_render.close()

# # === Save video ===
# video_path = "videos/flappy_bc_model.mp4"
# imageio.mimsave(video_path, frames, fps=30)
# print(f"🎥 Saved video to: {video_path}")

# # === Save actions ===
# action_path = "videos/flappy_bc_actions.pkl"
# with open(action_path, "wb") as f:
#     pickle.dump(actions, f)
# print(f"✅ Saved actions to: {action_path}")
