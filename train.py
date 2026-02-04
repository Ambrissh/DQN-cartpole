import torch
import os
from itertools import count
from config import Config
from utils import setup_environment, plot_durations
from agent import DQNAgent

def save_checkpoint(agent, episode, episode_durations, filename='latest.pth'):
    """Save training checkpoint"""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        'episode': episode,
        'policy_net': agent.policy_net.state_dict(),
        'target_net': agent.target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'steps_done': agent.steps_done,
        'episode_durations': episode_durations,
    }, os.path.join(Config.CHECKPOINT_DIR, filename))

def train():
    """Main training loop"""
    config = Config()
    env, device = setup_environment(config)
    
    # Get state/action dimensions
    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n
    
    # Create agent
    agent = DQNAgent(n_observations, n_actions, config, device)
    episode_durations = []
    
    print(f"Training on {device}")
    print(f"Episodes: {config.NUM_EPISODES}")
    
    for i_episode in range(config.NUM_EPISODES):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in count():
            # Select and perform action
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            
            # Observe new state
            next_state = None if done else torch.tensor(observation, dtype=torch.float32, 
                                                       device=device).unsqueeze(0)
            
            # Store transition
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            
            # Optimize
            agent.optimize_model()
            agent.soft_update_target_network()
            
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        
        # Save checkpoint
        if (i_episode + 1) % config.SAVE_FREQUENCY == 0:
            save_checkpoint(agent, i_episode + 1, episode_durations, 
                          f'checkpoint_ep{i_episode + 1}.pth')
            print(f"Episode {i_episode + 1}/{config.NUM_EPISODES} done. Checkpoint saved.")
    
    print('Training complete!')
    plot_durations(episode_durations, show_result=True)
    save_checkpoint(agent, config.NUM_EPISODES, episode_durations, 'final.pth')

if __name__ == "__main__":
    train()
