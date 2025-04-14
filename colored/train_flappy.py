# import gym
# import flappy_bird_gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# import torch

# # Reward shaping wrapper: small bonus each time step
# class SurvivalBonusWrapper(gym.RewardWrapper):
#     def __init__(self, env, bonus=0.1):
#         super(SurvivalBonusWrapper, self).__init__(env)
#         self.bonus = bonus

#     def reward(self, reward):
#         # Add survival bonus to base reward (e.g., for passing pipe)
#         return reward + self.bonus
# from stable_baselines3.common.callbacks import CheckpointCallback

# # Create a checkpoint callback that saves every 100,000 steps
# checkpoint_callback = CheckpointCallback(
#     save_freq=500_000,                   # Save every 100k steps
#     save_path="./checkpoints/",         # Directory to save models
#     name_prefix="flappy_checkpoint",    # Files will be flappy_checkpoint_<step>.zi
# )

# # # Then pass it to learn()
# # model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)


# # Check if GPU is available and set the device accordingly
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Wrap the FlappyBird env with the reward wrapper and vectorizer
# env = DummyVecEnv([lambda: SurvivalBonusWrapper(flappy_bird_gym.make("FlappyBird-v0"), bonus=0.1)])

# # Create PPO model
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_flappy_tensorboard/", device=device)

# # Train the model
# model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)  # you can start with 200k but 1M+ is better

# # Save model
# model.save("ppo_flappy3")


import os
import torch
import gym
import flappy_bird_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# ✅ Reward shaping wrapper
class SurvivalBonusWrapper(gym.RewardWrapper):
    def __init__(self, env, bonus=0.1):
        super(SurvivalBonusWrapper, self).__init__(env)
        self.bonus = bonus

    def reward(self, reward):
        # Only add bonus if the bird is still alive (i.e., not crashed)
        return reward + self.bonus if reward > 0 else reward

# ✅ Checkpointing every 500,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=500_000,
    save_path="./checkpoints/",
    name_prefix="flappy_checkpoint2",
)

# ✅ Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ Environment setup
env = DummyVecEnv([
    lambda: SurvivalBonusWrapper(flappy_bird_gym.make("FlappyBird-v0"), bonus=0.1)
])

# ✅ Custom policy architecture
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])]
)
resume_checkpoint = "checkpoints/flappy_checkpoint2_3000000_steps.zip"
if os.path.exists(resume_checkpoint):
    print(f"🔁 Resuming training from: {resume_checkpoint}")
    model = PPO.load(
        resume_checkpoint,
        env=env,
        tensorboard_log="./ppo_flappy_tensorboard/",
        device=device
    )
else:
    print("🆕 Starting new model from scratch.")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_flappy_tensorboard/",
        device=device,
        policy_kwargs=policy_kwargs
    )

# checkpoint_path = "checkpoints/flappy_checkpoint2_3000000_steps.zip"
# print(f"🔁 Resuming training from: {checkpoint_path}")

# model = PPO.load(
#     checkpoint_path,
#     env=env,                      # Reattach env
#     tensorboard_log="./ppo_flappy_tensorboard/",
#     device=device
# )

# # ✅ PPO model creation
# model = PPO(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./ppo_flappy_tensorboard/",
#     device=device,
#     policy_kwargs=policy_kwargs
# )

# ✅ Train model
total_timesteps = 5_000_000
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# ✅ Final save
model.save("ppo_flappy_final")
print(f"✅ Model saved after {total_timesteps:,} steps.")
