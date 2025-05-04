import gymnasium as gym
from gymnasium.utils.save_video import save_video
import a3_gym_env
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
import os
from datetime import datetime
import json
from tqdm import tqdm
from pend import PendulumEnv

import modules
from modules import Net, ReplayMemory, PolicyNet
from torch.distributions import MultivariateNormal

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Folders.
RESULTS = 'results'
if not os.path.exists(RESULTS):
    os.makedirs(RESULTS)
time_now = datetime.now().strftime('%Y%m%d-%H%M%S')
if not os.path.exists(os.path.join(RESULTS, time_now)):
    os.makedirs(os.path.join(RESULTS, time_now))
FOLDER_RESULTS = os.path.join(RESULTS, time_now)
ABS_FOLDER_RESUlTS = os.path.abspath(FOLDER_RESULTS)
print(f"Saving results to {ABS_FOLDER_RESUlTS}")

WARNING_EMOJI = "\u26A0\uFE0F"

# Set plot parameters.
import matplotlib
font = {'family' : 'serif',
#        'serif' : 'Computer Modern Roman',
        'size'   : 16}
matplotlib.rc('font', **font)
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams["figure.figsize"] = [3*3.54, 1.5*3.54]

# Set the seed.
torch.manual_seed(42)
np.random.seed(42)

# Set up logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up argument parser
parser = argparse.ArgumentParser(description='PPO Training on Pendulum Environment')
parser.add_argument('--clipping_on', action='store_true',
                    help='Enable clipping in PPO')
parser.add_argument('--advantage_on', action='store_true',
                    help='Enable advantage estimation in PPO')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                    help='Select mode: train or test')

# Hyperparameters
parser.add_argument('--num_timesteps_per_trajectory', type=int, default=200,
                    help='Number of timesteps per trajectory')
parser.add_argument('--num_trajectories', type=int, default=10,
                    help='Number of trajectories to collect')
parser.add_argument('--num_iterations', type=int, default=250,
                    help='Number of training iterations')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs per iteration')
parser.add_argument('--batch_size', type=int, default=10,
                    help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='Learning rate')
parser.add_argument('--eps', type=float, default=0.2,
                    help='Clipping parameter')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='Discount factor')
parser.add_argument('--lambda_', type=float, default=0.95,
                    help='GAE parameter')
parser.add_argument('--std', type=float, default=0.1,
                    help='Standard deviation for action sampling')
# For experiments with spectral normalization.
# Specific to the loss of plasticity paper.
parser.add_argument('--do_loss_of_plasticity', action='store_true',
                    help='Do loss of plasticity')

parser.add_argument('--use_spectral_norm', action='store_true',
                    help='Use spectral normalization')
parser.add_argument('--load_file', type=str, default=None,
                    help='Specify a full path to a specific .pt file to load policy network weights from (this takes precedence over folder_time)')


args = parser.parse_args()
# Save the arguments.
with open(ABS_FOLDER_RESUlTS + '/args.json', 'w') as f:
    json.dump(args.__dict__, f)

# Params env. (they don't need to change via cmd line)
PARAMS_ENV_LOP = {
    'MASS_UPPER_BOUND': 1.5, # Mass
    'MASS_LOWER_BOUND': 1.2,
    'CHANGE_MASS': True,
    'LENGTH_UPPER_BOUND': 1.5, # Length
    'LENGTH_LOWER_BOUND': 0.5,
    'CHANGE_LENGTH': False,
    'DAMPING_UPPER_BOUND': 1.0, # Damping
    'DAMPING_LOWER_BOUND': 0.4,
    'CHANGE_DAMPING': True,
    'TIME_TO_CHANGE_ENV': 40/100 * args.num_iterations,
    'CHANGE_ENV_INTERVAL': 20,
}
# Save the params env.
with open(ABS_FOLDER_RESUlTS + '/params_env_lop.json', 'w') as f:
    json.dump(PARAMS_ENV_LOP, f)
# Print the params env.
print(f"Params env: {PARAMS_ENV_LOP}")
POLICY_FOLDER = '/policy_net'
CRITIC_FOLDER = '/critic_net'
if not os.path.exists(ABS_FOLDER_RESUlTS + POLICY_FOLDER):
    os.makedirs(ABS_FOLDER_RESUlTS + POLICY_FOLDER)
if not os.path.exists(ABS_FOLDER_RESUlTS + CRITIC_FOLDER):
    os.makedirs(ABS_FOLDER_RESUlTS + CRITIC_FOLDER)

# Create the custom environment.
gym.register(
    id="lop/pend-v0",
    entry_point=PendulumEnv,
)

env = gym.make('lop/pend-v0', target=np.pi / 8)
env = env.unwrapped  # Get the unwrapped environment to access custom methods.

# Hyperparameters.
num_timesteps_per_trajectory = args.num_timesteps_per_trajectory
num_trajectories = args.num_trajectories
num_iterations = args.num_iterations
epochs = args.epochs

batch_size = args.batch_size
learning_rate = args.learning_rate
eps = args.eps # clipping parameter.
gamma = args.gamma
lambda_ = args.lambda_
std = args.std

def calc_reward_togo(rewards, gamma=0.99):
    ''' Calculate the (discounted) reward-to-go from a sequence of rewards. '''
    n = len(rewards)
    reward_togo = np.zeros(n)
    reward_togo[-1] = rewards[-1]
    for i in reversed(range(n-1)):
        reward_togo[i] = rewards[i] + gamma * reward_togo[i+1]

    reward_togo = torch.tensor(reward_togo, dtype=torch.float)
    return reward_togo

def calc_advantages(rewards, values, gamma=0.99, lambda_=1):
    ''' Calculate the advantages from a sequence of rewards and values. '''
    advantages = torch.zeros_like(torch.as_tensor(rewards))
    sum = 0
    for t in reversed(range(len(rewards)-1)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        sum = delta + gamma * lambda_ * sum
        advantages[t] = sum
    return advantages


class PPO:
    def __init__(self, gamma=0.99):

        env_input_size = env.observation_space.shape[0]
        env_action_size = env.action_space.shape[0]

        if args.use_spectral_norm:
            self.policy_net = PolicyNet(env_input_size, env_action_size).to(device)
        else:
            self.policy_net = Net(env_input_size, env_action_size, hidden_size=7).to(device)
            print(f"Number of parameters in policy net: {sum(p.numel() for p in self.policy_net.parameters())}")

        self.critic_net = Net(env_input_size, 1).to(device)

        self.optimizer = torch.optim.Adam([  # Update both models together
            {'params': self.policy_net.parameters(), 'lr': learning_rate},
            {'params': self.critic_net.parameters(), 'lr': learning_rate}
                    ])

        self.memory = ReplayMemory(batch_size)

        self.gamma = gamma
        self.lambda_ = 1
        self.vf_coef = 1  # c1
        self.entropy_coef = 0.01  # c2

        self.clipping_on = args.clipping_on
        self.advantage_on = args.advantage_on

        self.std = torch.diag(torch.full(size=(1,), fill_value=0.5)).to(device)

    def generate_trajectory(self, render: bool = False):

        current_state, _ = env.reset()  # Gymnasium returns (state, info)
        states = []
        actions = []
        rewards = []
        log_probs = []

        # Run the old policy in environment for num_timestep.
        for t in range(num_timesteps_per_trajectory):

            mean = self.policy_net(torch.as_tensor(current_state).to(device))

            # Sample an action.
            normal = MultivariateNormal(mean, self.std)
            action = normal.sample().detach()
            log_prob = normal.log_prob(action).detach()

            # Emulate taking that action.
            next_state, reward, _, _, _ = env.step(action.cpu().numpy())  # Gymnasium returns (state, reward, terminated, truncated, info)

            # Store results in a list.
            states.append(current_state)
            actions.append(action.cpu())
            rewards.append(reward)
            log_probs.append(log_prob.cpu())

            if render:
                env.render()

            current_state = next_state.copy()

        # Calculate reward to go.
        rtg = calc_reward_togo(torch.as_tensor(rewards), self.gamma).to(device)

        # Calculate values.
        values = self.critic_net(torch.as_tensor(states).to(device)).squeeze()

        # Calculate advantages.
        advantages = calc_advantages(rewards, values.detach().cpu(), self.gamma, self.lambda_).to(device)

        # Save the transitions in replay memory.
        for t in range(len(rtg)):
            self.memory.push(states[t], actions[t], rewards[t], rtg[t], advantages[t], values[t], log_probs[t])

    def train(self):
        
        train_actor_loss = []
        train_critic_loss = []
        train_total_loss = []
        train_reward = []

        time_to_change_env_list = []
        for idx_iteration in tqdm(range(num_iterations), desc="Training Progress"):

            # Collect a number of trajectories and save the transitions in replay memory.
            # Some trajectories will have an altered environment.
            for idx_trajectory in range(num_trajectories):

                # Change the environment.
                if args.do_loss_of_plasticity:
                    if idx_trajectory > np.floor(num_trajectories / 2):
                        if idx_iteration >= PARAMS_ENV_LOP['TIME_TO_CHANGE_ENV']: # Assume that the reward is stable after 60% of the iterations.
                            # Change the environment.
                            if idx_iteration % PARAMS_ENV_LOP['CHANGE_ENV_INTERVAL'] == 0:
                                print(f"{WARNING_EMOJI} Changing the environment.")
                                # if PARAMS_ENV_LOP['CHANGE_DAMPING']:
                                #     env.set_damping(b=np.random.uniform(PARAMS_ENV_LOP['DAMPING_LOWER_BOUND'], PARAMS_ENV_LOP['DAMPING_UPPER_BOUND']))
                                # if PARAMS_ENV_LOP['CHANGE_MASS']:
                                #     env.set_mass(m=np.random.uniform(PARAMS_ENV_LOP['MASS_LOWER_BOUND'], PARAMS_ENV_LOP['MASS_UPPER_BOUND']))
                                # if PARAMS_ENV_LOP['CHANGE_LENGTH']:
                                #     env.set_length(l=np.random.uniform(PARAMS_ENV_LOP['LENGTH_LOWER_BOUND'], PARAMS_ENV_LOP['LENGTH_UPPER_BOUND']))
                                # PARAMS_ENV_LOP['TIME_TO_CHANGE_ENV'] = idx_iteration + PARAMS_ENV_LOP['CHANGE_ENV_INTERVAL']
                                # time_to_change_env_list.append(PARAMS_ENV_LOP['TIME_TO_CHANGE_ENV'])
                                env.set_target(- np.pi / 8)

                self.generate_trajectory()

            # Sample from replay memory.
            states, actions, rewards, rewards_togo, advantages, values, log_probs, batches = self.memory.sample()

            actor_loss_list = []
            critic_loss_list = []
            total_loss_list = []
            reward_list = []
            for _ in range(epochs):

                # Calculate the new log prob.
                mean = self.policy_net(states)
                normal = MultivariateNormal(mean, self.std)
                new_log_probs = normal.log_prob(actions.unsqueeze(-1))

                r = torch.exp(new_log_probs - log_probs)
                
                if self.clipping_on == True:
                    clipped_r = torch.clamp(r, 1 - eps, 1 + eps)
                else:
                    clipped_r = r

                new_values = self.critic_net(states).squeeze()
                returns = (advantages + values).detach()

                if self.advantage_on == True:
                    actor_loss = (-torch.min(r * advantages, clipped_r * advantages)).mean()
                    critic_loss = nn.MSELoss()(new_values.float(), returns.float())
                else:
                    actor_loss = (-torch.min(r * rewards_togo, clipped_r * rewards_togo)).mean()
                    critic_loss = nn.MSELoss()(new_values.float(), rewards_togo.float())

                # Calculate total loss.
                total_loss = actor_loss + (self.vf_coef * critic_loss) - (self.entropy_coef * normal.entropy().mean())

                # Update policy and critic network.
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                reward_list.append(sum(rewards))

            self.memory.clear()

            avg_actor_loss = sum(actor_loss_list) / len(actor_loss_list)
            avg_critic_loss = sum(critic_loss_list) / len(critic_loss_list)
            avg_total_loss = sum(total_loss_list) / len(total_loss_list)
            avg_reward = sum(reward_list) / len(reward_list)

            train_actor_loss.append(avg_actor_loss)
            train_critic_loss.append(avg_critic_loss)
            train_total_loss.append(avg_total_loss)
            train_reward.append(avg_reward)

            tqdm.write(f'Actor loss = {avg_actor_loss:.4f} | Critic loss = {avg_critic_loss:.4f} | Total Loss = {avg_total_loss:.4f} | Reward = {avg_reward:.4f}')

            # Save the networks.
            torch.save(self.policy_net.state_dict(), ABS_FOLDER_RESUlTS + POLICY_FOLDER + f'/policy_net' + str(idx_iteration) + '.pt')
            torch.save(self.policy_net.state_dict(), ABS_FOLDER_RESUlTS + POLICY_FOLDER + f'/policy_net.pt')
            torch.save(self.critic_net.state_dict(), ABS_FOLDER_RESUlTS + CRITIC_FOLDER + f'/critic_net' + str(idx_iteration) + '.pt')

            # Plot the performance.
            fig = plt.figure()
            axes = []
            for i in range(3):
                axes.append(plt.subplot2grid((2, 3), (0, i)))
            reward_ax = plt.subplot2grid((2, 3), (1, 0), colspan=3)
            axes[0].plot(range(len(train_actor_loss)), [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_actor_loss], 'r', label='Actor Loss')
            axes[0].set_title('Actor Loss')
            axes[1].plot(range(len(train_critic_loss)), [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_critic_loss], 'b', label='Critic Loss')
            axes[1].set_title('Critic Loss')
            axes[2].plot(range(len(train_total_loss)), [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in train_total_loss], 'm', label='Total Loss')
            axes[2].set_title('Total Loss')
            for i in range(3):
                axes[i].set_xlabel('Iteration')
            # Plot reward spanning all columns
            reward_ax.plot(range(len(train_reward)), [reward.cpu().item() if torch.is_tensor(reward) else reward for reward in train_reward], 'orange', label='Accumulated Reward')
            if args.do_loss_of_plasticity:
                for time_to_change_env in time_to_change_env_list:
                    reward_ax.axvline(x=time_to_change_env, color='black', linestyle='--', label='Time to change environment', alpha=0.1)
            # Horizontal line at y=0.
            reward_ax.axhline(y=0, color='green', linestyle='-', label='Zero Reward')
            reward_ax.set_title('Accumulated Reward')
            reward_ax.set_xlabel('Iteration')

            plt.tight_layout()
            plt.savefig(ABS_FOLDER_RESUlTS + f'/performance.png')
            plt.close(fig)
        self.show_value_grid()
    
    def show_value_grid(self):

        # Sweep theta and theta_dot and find all states.
        theta = torch.linspace(-np.pi, np.pi, 100)
        theta_dot = torch.linspace(-8, 8, 100)
        values = torch.zeros((len(theta), len(theta_dot)))

        for i, t in enumerate(theta):
            for j, td in enumerate(theta_dot):
                state = (torch.cos(t), torch.sin(t), td)
                values[i, j] = self.critic_net(torch.as_tensor(state).to(device))

        fig = plt.figure()
        plt.imshow(values.cpu().detach().numpy(), extent=[theta[0], theta[-1], theta_dot[0], theta_dot[-1]] ,aspect=0.4)
        plt.title('Value grid')
        plt.xlabel('angle')
        plt.ylabel('angular velocity')
        plt.savefig(ABS_FOLDER_RESUlTS + f'/value_grid.png')
        plt.close(fig)

    def test(self):
        if args.load_file:
            policy_path = args.load_file
        else:
            policy_path = ABS_FOLDER_RESUlTS + POLICY_FOLDER + '/policy_net.pt'
        # Load the network weights.
        self.policy_net.load_state_dict(torch.load(policy_path, weights_only=True))

        current_state, _ = env.reset()  # Gymnasium returns (state, info)
        reward_list = []
        print("Testing the policy...")
        frames = []
        for i in range(200):
            mean = self.policy_net(torch.as_tensor(current_state).to(device))
            normal = MultivariateNormal(mean, self.std)
            action = normal.sample().cpu().detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)  # Gymnasium returns (state, reward, terminated, truncated, info).

            current_state = next_state.copy()
            reward_list.append(reward)

            # env.render()
            frames.append(env.render(mode="rgb_array"))
        
        save_video(
            frames=frames,
            video_folder="videos",
            fps=env.metadata["render_fps"],
        )
        avg_reward = sum(reward_list) / len(reward_list)
        print(f"Average reward: {avg_reward}")
        env.close()
 
if __name__ == '__main__':

    # Print parser arguments.
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    agent = PPO()

    if args.mode == 'train':
        agent.train()

    agent.test()
