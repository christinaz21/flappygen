# import os
# os.environ["SDL_AUDIODRIVER"] = "dummy"
# os.environ["SDL_VIDEODRIVER"] = "dummy"

# import gym
# import numpy as np
# import imageio
# from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB

# os.makedirs("videos", exist_ok=True)

# env = FlappyBirdEnvRGB(bird_color="yellow", background="night")

# frames = []
# obs = env.reset()
# done = False
# frame_count = 0

# while not done:
#     # Jump every 6 frames, otherwise do nothing
#     action = 1 if frame_count % 6 == 0 else 0
#     frame_count += 1

#     obs, reward, done, info = env.step(action)
#     frame = env.render(mode="rgb_array")
#     frame = np.transpose(frame, (1, 0, 2))  # Rotate if needed
#     frames.append(frame)

# env.close()

# output_path = os.path.join("videos", "flappy_bird_night.mp4")
# imageio.mimsave(output_path, frames, fps=30)

# print(f"Saved video to {output_path}")


import os
import numpy as np
import imageio
import pickle
import torch
# import flappy_bird_gym
# from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
from stable_baselines3 import PPO
import sys
sys.path.append("flappy-bird-gym")
import flappy_bird_gym

from flappy_bird_gym.envs.flappy_bird_env_rgb import FlappyBirdEnvRGB
# Setup
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.makedirs("videos", exist_ok=True)

# Load model
model = PPO.load("checkpoints/flappy_checkpoint2_3000000_steps.zip", device="cpu")

# Create policy env (for done) and RGB env (for rendering)
env_policy = flappy_bird_gym.make("FlappyBird-v0")
env_render = FlappyBirdEnvRGB(bird_color="black", background="day")

obs_policy = env_policy.reset()
env_render.reset()

frames = []
actions = []

last_frame = None
unchanged_count = 0
unchanged_threshold = 5  # Stop if same frame repeats this many times

step = 0
while True:
    action, _ = model.predict(obs_policy)
    actions.append(torch.tensor(int(action)))

    # Step policy env
    obs_policy, reward, done, info = env_policy.step(action)

    # Step and render RGB env
    env_render.step(action)
    frame = env_render.render(mode="rgb_array")
    frame = np.transpose(frame, (1, 0, 2))

    # Check if frame is unchanged
    if last_frame is not None and np.array_equal(frame, last_frame):
        unchanged_count += 1
    else:
        unchanged_count = 0

    last_frame = frame.copy()
    frames.append(frame)

    if unchanged_count >= unchanged_threshold:
        print(f"🧊 Frame frozen at step {step} — exiting.")
        break

    if done:
        print(f"🛑 Env done=True at step {step}")
        break

    step += 1

# Close envs
env_policy.close()
env_render.close()

# Save video
video_path = "videos/flappy_trained2.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"🎥 Saved video to: {video_path}")

# Save action list
action_path = "videos/flappy_trained_actions.pkl"
with open(action_path, "wb") as f:
    pickle.dump(actions, f)
print(f"✅ Saved actions to: {action_path}")

