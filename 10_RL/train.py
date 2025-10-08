import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from dotenv import load_dotenv
import os
import ale_py
import time

# Load environment variables
load_dotenv()

# Set SDL environment variables for WSL2 (same as main.py)
os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
# Try to set a smaller zoom factor for the Atari rendering
os.environ['ALE_DISPLAY_SCREEN_ZOOM'] = '2.0'  # Default is often 4.0, this makes it smaller

class PPONetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PPONetwork, self).__init__()
        
        # CNN layers for image processing
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)
        
        # Shared layers
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(512, num_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(512, 1)
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        x = x.float() / 255.0  # Normalize pixel values
        conv_out = self.conv(x).view(x.size()[0], -1)
        shared = self.fc_shared(conv_out)
        
        action_probs = torch.softmax(self.actor(shared), dim=-1)
        state_value = self.critic(shared)
        
        return action_probs, state_value

class PPOTrainer:
    def __init__(self, env, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=4):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        
        # Get environment info
        self.num_actions = env.action_space.n
        self.input_shape = (4, 84, 84)  # Assuming frame stacking
        
        # Initialize network
        self.network = PPONetwork(self.input_shape, self.num_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Storage for experiences
        self.reset_storage()
    
    def reset_storage(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def preprocess_state(self, state):
        # Convert to grayscale and resize
        if len(state.shape) == 3:
            state = np.mean(state, axis=2)
        state = np.array(state, dtype=np.uint8)
        # Resize to 84x84 (you might need to install cv2 or use PIL)
        # For now, just return as is - you should add proper preprocessing
        return state
    
    def collect_experience(self, num_steps=2048):
        state, _ = self.env.reset()
        state = self.preprocess_state(state)
        
        for step in range(num_steps):
            # Stack frames (simplified - you should implement proper frame stacking)
            stacked_state = np.stack([state] * 4, axis=0)
            state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0)
            
            with torch.no_grad():
                action_probs, value = self.network(state_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            # Store experience
            self.states.append(stacked_state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(value.item())
            
            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            
            self.rewards.append(reward)
            self.dones.append(done)
            
            state = self.preprocess_state(next_state)
            
            if done:
                state, _ = self.env.reset()
                state = self.preprocess_state(state)
    
    def compute_advantages(self):
        returns = []
        advantages = []
        
        # Compute returns (discounted rewards)
        discounted_return = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        # Compute advantages
        returns = torch.FloatTensor(returns)
        values = torch.FloatTensor(self.values)
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update_policy(self, returns, advantages):
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        for _ in range(self.epochs):
            # Forward pass
            action_probs, values = self.network(states)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            # Compute ratios
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # Compute policy loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()
    
    def train(self, num_iterations=1000):
        for iteration in range(num_iterations):
            # Collect experience
            self.collect_experience()
            
            # Compute advantages
            returns, advantages = self.compute_advantages()
            
            # Update policy
            policy_loss, value_loss = self.update_policy(returns, advantages)
            
            # Reset storage
            self.reset_storage()
            
            # Print progress
            if iteration % 10 == 0:
                avg_reward = np.mean(self.rewards) if self.rewards else 0
                print(f"Iteration {iteration}, Avg Reward: {avg_reward:.2f}, "
                      f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            
            # Save model periodically
            if iteration % 20 == 0:
                torch.save(self.network.state_dict(), f'mario_ppo_model_{iteration}.pth')

def main():
    try:
        # Create environment (same as main.py but without rendering for training)
        print("Creating Mario Bros environment for training...")
        env = gym.make("ALE/MarioBros-v5", render_mode=None)  # No rendering for training
        
        # Create trainer
        trainer = PPOTrainer(env)
        
        print("Starting PPO training for Mario Bros...")
        print(f"Action space: {env.action_space.n} actions")
        print(f"Observation space: {env.observation_space.shape}")
        print("\nAction mapping for Mario Bros:")
        print("0: NOOP, 1: Fire, 2: Up, 3: Right, 4: Left, 5: Down")
        print("6: Up-Right, 7: Up-Left, 8: Down-Right, 9: Down-Left")
        print("10: Up-Fire, 11: Right-Fire, 12: Left-Fire, 13: Down-Fire")
        print("14: Up-Right-Fire, 15: Up-Left-Fire, 16: Down-Right-Fire, 17: Down-Left-Fire")
        
        # Start training
        trainer.train(num_iterations=100)  # Reduced for testing
        
        # Save final model
        torch.save(trainer.network.state_dict(), 'mario_ppo_final.pth')
        print("Training completed! Model saved as 'mario_ppo_final.pth'")
        
        env.close()
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
