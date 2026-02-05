
# DQN CartPole

Deep Q-Network (DQN) implementation for solving Gymnasium's CartPole-v1 environment.

 Project Overview

This project implements a DQN agent that learns to balance a pole on a moving cart. The agent achieves perfect performance (500/500 steps) by learning an optimal policy through deep reinforcement learning.

## Project Structure
```
dqn-cartpole/
├── config.py           # Hyperparameters and configuration
├── model.py            # DQN neural network architecture
├── replay_buffer.py    # Experience replay buffer
├── agent.py            # DQN agent with training logic
├── utils.py            # Utility functions (plotting, setup)
├── checkpoint.py       # Checkpoint saving/loading
├── inference.py        # Model testing and evaluation
├── train.py            # Main training script
├── checkpoints/        # Saved model checkpoints
└── README.md           # This file
```
##  Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Ambrissh/DQN-cartpole.git
cd DQN-cartpole

# Install dependencies
pip install torch gymnasium matplotlib
```

### Training
```bash
# Train from scratch, python train.py

# Resume from checkpoint (if exists)
python train.py  # Automatically resumes from checkpoints/latest.pth
```

### Testing/Inference

**Option 1: Python script**
```bash
# Test with default checkpoint (10 episodes, no rendering)
python inference.py

# Test with custom checkpoint
python inference.py --checkpoint checkpoints/final.pth --episodes 20

# Visual demo (watch the agent play)
python inference.py --checkpoint checkpoints/latest.pth --episodes 3 --render
```

**Option 2: Python code**
```python
from inference import ModelTester

# Load and test model
tester = ModelTester('checkpoints/latest.pth')
rewards = tester.test(num_episodes=10, render=False)

# Or use convenience functions
from inference import quick_test, demo

quick_test('checkpoints/latest.pth', episodes=10)  # Quick test
demo('checkpoints/latest.pth', episodes=3)         # Visual demo
```

**Option 3: Jupyter/Colab**
```python
# In a notebook cell
%run inference.py --checkpoint checkpoints/latest.pth --episodes 10
```

##  Results

The trained agent achieves:
- **Average reward**: 500.0 (maximum possible)
- **Success rate**: 100%
- **Training episodes**: 600 (GPU) / 50 (CPU)

##  Model Architecture

- **Input**: 4 observations (cart position, velocity, pole angle, angular velocity)
- **Hidden layers**: 2 x 128 neurons with ReLU activation
- **Output**: 2 Q-values (one per action: left/right)

## ⚙️ Hyperparameters
```python
BATCH_SIZE = 128
GAMMA = 0.99          # Discount factor
EPS_START = 0.9       # Initial exploration
EPS_END = 0.01        # Final exploration
EPS_DECAY = 2500      # Exploration decay rate
TAU = 0.005           # Soft update rate
LEARNING_RATE = 3e-4  # AdamW optimizer
```

## Pre-trained Models

Download pre-trained checkpoints from the `checkpoints/` directory:
- `latest.pth` - Most recent training checkpoint
- `final.pth` - Final trained model (if training completed)
- `checkpoint_epXXX.pth` - Intermediate checkpoints (every 50 episodes)

##  Customization

### Train in a different environment
```python
# In config.py
ENV_NAME = 'LunarLander-v2'  # Change environment
```

### Modify network architecture
```python
# In model.py
self.layer1 = nn.Linear(n_observations, 256)  # Increase size
```

### Change training duration
```python
# In config.py
NUM_EPISODES = 1000  # Train longer
```

## Code Usage

### Load a trained model
```python
from inference import ModelTester

tester = ModelTester('checkpoints/latest.pth')
rewards = tester.test(num_episodes=10)
```

### Save/Load checkpoints
```python
from checkpoint import CheckpointManager

manager = CheckpointManager()
manager.save(agent, episode=100, episode_durations=[...])
episode, durations = manager.load(agent, filename='latest.pth')
```

### Custom testing
```python
from inference import ModelTester

tester = ModelTester('checkpoints/latest.pth')

# Test with custom parameters
rewards = tester.test(
    env_name='CartPole-v1',
    num_episodes=50,
    render=True,
    verbose=True
)
```

##  Troubleshooting

**CUDA errors**: Set device to CPU
```python
# In config or inference
device = torch.device('cpu')
```

**Import errors**: Ensure all files are in the same directory

**No checkpoints found**: Train the model first with `python train.py`

##  References

- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [OpenAI Gym](https://gymnasium.farama.org/)

##  License

MIT License

##  Author

Ambrissh - [GitHub](https://github.com/Ambrissh)

---

**Star this repo if you found it helpful!**
