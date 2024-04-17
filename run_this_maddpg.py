import torch
import numpy as np
from robot_motion_env import RobotMotionEnv
from torch.utils.tensorboard import SummaryWriter
import argparse
from MADDPG.replay_buffer import ReplayBuffer
from MADDPG.maddpg import MADDPG
# from MADDPG.matd3 import MATD3
import copy

np.random.seed(2)

if __name__ == '__main__':
    env = RobotMotionEnv("UAV-0", False)

    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    # parser.add_argument("--N", type=int, default=3, help=" number of agents")

    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5, help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    args = parser.parse_args()
    agent_num = 3
    args.N = agent_num
    args.obs_dim_n = [env.state_dim for i in range(args.N)]
    args.action_dim_n = [env.action_dim for i in range(args.N)]
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    agents = []
    for i in range(agent_num):
        agents.append(MADDPG(args, i))
    ctrl = agents
    env.after_mainloop(display=False, mode="train", episode=2000, predict=1000, ctrl=ctrl, args=args)
    # env.after_mainloop(display=True, mode="test", episode=5000, predict=1000, ctrl=ctrl, args=args)
    # print("quit program!")

