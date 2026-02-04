import torch
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib
from IPython import display

def setup_environment(config):
    """Initialize environment and set seeds"""
    env = gym.make(config.ENV_NAME)
    
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    env.reset(seed=config.SEED)
    env.action_space.seed(config.SEED)
    env.observation_space.seed(config.SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    return env, device

def plot_durations(episode_durations, show_result=False):
    """Plot training progress"""
    is_ipython = 'inline' in matplotlib.get_backend()
    
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())
    
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
