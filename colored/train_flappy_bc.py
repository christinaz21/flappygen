import sys
sys.path.append("flappy-bird-gym")
import flappy_bird_gym

import numpy as np
import torch
from torch import nn
from torch.distributions import Distribution
import torch.backends.cudnn
torch.backends.cudnn.benchmark = True

def evaluate_policy(env, policy, discount) -> float:
  """Returns a single-sample estimate of a policy's return.

  Args:
    env: OpenAI gym environment following old API.
    policy: Callable representing a policy.

  Returns:
    Single-sample estimate of return.
  """
  ### YOUR CODE HERE!
  pass
  state = env.reset()
  done = False
  total_reward = 0.0
  t = 0

  while not done:
      state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
      action_dist = policy(state_tensor)
      action = action_dist.sample()
      next_state, reward, done, _ = env.step(action.item())

      total_reward += reward * (discount ** t)
      state = next_state
      t += 1

  return total_reward


class DiscretePolicy(nn.Module):
  def __init__(self, input_dim, n_actions):
    """Initializes a policy over the action set {0, 1, ..., n_actions-1}.

    Args:
      input_dim: Observation dimensionality.
      n_actions: Number of actions in environment.
    """
    super().__init__()
    ### YOUR CODE HERE!
    pass
    # two-layer neural network
    self.linear1 = nn.Linear(input_dim,32)
    self.linear2 = nn.Linear(32, n_actions)

  def forward(self, ob) -> Distribution:
    """Returns a distribution over this policy's action set.
    """
    ### YOUR CODE HERE!
    pass
    x = nn.functional.relu(self.linear1(ob))
    x = self.linear2(x)
    return torch.distributions.Categorical(logits=x)


# def train_policy_by_bc(policy, dataset, n_steps, batch_size) -> DiscretePolicy:
#   """Trains the provided policy by behavioral cloning, by taking n_steps training steps with the
#   provided optimizer. During training, training batches of size batch_size are sampled from the dataset
#   to compute the loss.

#   Args:
#     policy: policy of class DiscretePolicy.
#     dataset: The dataset, represented as TensorDataset(observations, actions), where observations
#               and actions are both tensors from the original dataset
#     n_steps: Number of training steps to take.
#     batch_size: Size of the sampled batch for each training step.

#   Returns:
#     A policy trained according to the parameters above.
#   """
#   ### YOUR CODE HERE!
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
#   criterion = nn.CrossEntropyLoss()

#   train_loader = torch.utils.data.DataLoader(
#       dataset= dataset,
#       batch_size=batch_size,
#       shuffle=True
#   )

#   policy.train()
#   for step in range(n_steps):
#       for batch_states, batch_actions in train_loader:
#           batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)
#           optimizer.zero_grad()
#           action_dist = policy(batch_states)
#           logits = action_dist.logits

#           loss = criterion(logits, batch_actions)

#           loss.backward()
#           optimizer.step()

#   return policy


from torch.cuda.amp import autocast, GradScaler

def train_policy_by_bc(policy, dataset, n_steps, batch_size, use_amp=True) -> DiscretePolicy:
    """Trains the provided policy by behavioral cloning with optional AMP.

    Args:
        policy: policy of class DiscretePolicy.
        dataset: TensorDataset(observations, actions)
        n_steps: Number of gradient steps to take.
        batch_size: Batch size per gradient step.
        use_amp: Whether to use automatic mixed precision (default: True).

    Returns:
        Trained policy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    scaler = GradScaler(enabled=use_amp)

    policy.train()
    policy.to(device)

    for step in range(n_steps):
        for batch_states, batch_actions in train_loader:
            batch_states, batch_actions = batch_states.to(device), batch_actions.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                action_dist = policy(batch_states)
                logits = action_dist.logits
                loss = criterion(logits, batch_actions)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if step % 10 == 0 or step == n_steps - 1:
            print(f"[Step {step+1}/{n_steps}] Loss: {loss.item():.4f}")

    return policy


### YOUR CODE HERE
pass
import scipy.io
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    data = scipy.io.loadmat('flappy_sr_notes.mat')
    observations = torch.tensor(data['observations'], dtype=torch.float32)
    actions = torch.tensor(data['actions'].flatten(), dtype=torch.long)
    dataset = TensorDataset(observations, actions)

    # define policy
    input_dim = observations.shape[1] # number of features
    n_actions = len(torch.unique(actions)) # number of actions
    policy = DiscretePolicy(input_dim, n_actions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # trained_policy = train_policy_by_bc(
    #     policy=policy,
    #     dataset=dataset,
    #     n_steps=50,
    #     batch_size=64
    # )

    trained_policy = train_policy_by_bc(
        policy=policy,
        dataset=dataset,
        n_steps=100,
        batch_size=64,     # or 512 if your GPU can handle it
        use_amp=True        # Set False if you're on older GPUs
    )

    env = flappy_bird_gym.make("FlappyBird-v0")

    # define a callable policy to use for evaluation
    class TrainedPolicy:
        def __init__(self, policy):
            self.policy = policy
            self.device = next(policy.parameters()).device  # Automatically detect device

        def __call__(self, state):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.policy(state_tensor)


    # evaluate trained policy
    discount = 0.999
    trained_returns = [evaluate_policy(env, TrainedPolicy(trained_policy), discount) for _ in range(50)]
    mean_trained_return = np.mean(trained_returns)

    # evaluate random policy
    # random_returns = [evaluate_policy(env, RandomPolicy(), discount) for _ in range(50)]
    # mean_random_return = np.mean(random_returns)

    print(f"Mean return (Trained policy): {mean_trained_return}")
    # print(f"Mean return (Random policy): {mean_random_return}")

    torch.save(trained_policy.state_dict(), "bc_trained_policy.pth")
    print("✅ Trained policy saved as bc_trained_policy.pth")

