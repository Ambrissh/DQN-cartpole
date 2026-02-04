from collections import deque
import random

class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
