import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import numpy as np
import imageio

import sys
sys.path.append("flappy-bird-gym")

import flappy_bird_gym

from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB

os.makedirs("videos", exist_ok=True)

env = FlappyBirdEnvRGB(bird_color="pink", background="night")

frames = []
obs = env.reset()
done = False
frame_count = 0

while not done:
    # Jump every 6 frames, otherwise do nothing
    action = 1 if frame_count % 6 == 0 else 0
    frame_count += 1

    obs, reward, done, info = env.step(action)
    frame = env.render(mode="rgb_array")
    frame = np.transpose(frame, (1, 0, 2))  # Rotate if needed
    frames.append(frame)

env.close()

output_path = os.path.join("videos", "flappy_bird_pink.mp4")
imageio.mimsave(output_path, frames, fps=30)

print(f"Saved video to {output_path}")

