# test_mario_model.py
import gymnasium as gym
import torch
import ale_py
import numpy as np
import cv2
import os
import time
from train import PPONetwork

# Performance optimization settings
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.set_num_threads(4)  # Optimize CPU usage

# Set SDL environment variables (same as main.py and train.py)
os.environ['SDL_VIDEO_WINDOW_POS'] = '100,100'
os.environ['ALE_DISPLAY_SCREEN_ZOOM'] = '2.0'

def preprocess_state(state):
    """Optimized preprocessing - same as training but faster"""
    if len(state.shape) == 3:
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_LINEAR)  # Faster interpolation
    return state.astype(np.uint8)

def get_speed_mode():
    """Interactive speed selection"""
    print("\n⚡ Choose playback speed:")
    print("1. 🐌 SLOW - Easy to watch (0.05s delay)")
    print("2. 🚀 FAST - Quick playback (0.01s delay)")  
    print("3. ⚡ ULTRA - Maximum speed (no delay)")
    print("4. 🎯 TURBO - Skip frames for analysis")
    
    choice = input("Enter choice (1-4) [default: 2]: ").strip() or "2"
    
    modes = {
        "1": ("slow", 0.05, 1),
        "2": ("fast", 0.01, 1), 
        "3": ("ultra", 0.0, 1),
        "4": ("turbo", 0.0, 3)  # Skip 2 out of 3 frames
    }
    
    return modes.get(choice, modes["2"])

def load_and_test_model():
    try:
        # Load the trained model with optimizations
        model = PPONetwork((4, 84, 84), 18)
        
        # Enable optimization modes for faster inference
        torch.set_grad_enabled(False)  # Disable gradient computation globally
        if torch.cuda.is_available():
            model = model.cuda()  # Use GPU if available
            print("🚀 Using GPU acceleration!")
        else:
            print("💻 Using CPU inference")
        
        # Try to load different model checkpoints
        model_files = [
            'mario_ppo_final.pth',
            'mario_ppo_model_80.pth', 
            'mario_ppo_model_60.pth',
            'mario_ppo_model_40.pth',
            'mario_ppo_model_20.pth',
            'mario_ppo_model_0.pth'
        ]
        
        model_loaded = False
        for model_file in model_files:
            try:
                if os.path.exists(model_file):
                    print(f"Loading model: {model_file}")
                    model.load_state_dict(torch.load(model_file))
                    model.eval()
                    model_loaded = True
                    break
            except Exception as e:
                print(f"Failed to load {model_file}: {e}")
                continue
        
        if not model_loaded:
            print("❌ No trained model found! Please run training first.")
            print("Available files:")
            for f in os.listdir('.'):
                if f.endswith('.pth'):
                    print(f"  - {f}")
            return
        
        # Create environment with rendering
        env = gym.make("ALE/MarioBros-v5", render_mode="human")
        
        print("🤖 Starting FAST AI Mario Bros Test!")
        print("⚡ Optimized for speed - The AI will play much faster!")
        print("🎯 Watch for rapid decision making and movements...")
        print("\n🎮 Speed Controls:")
        print("  - Press 'S' during play to toggle slow mode")
        print("  - Press 'F' for ultra-fast mode")
        print("\nAction meanings:")
        print("0: NOOP, 1: Fire, 2: Up, 3: Right, 4: Left, 5: Down")
        print("6: Up-Right, 7: Up-Left, 8: Down-Right, 9: Down-Left")
        print("10+: Fire combinations")
        
        # Interactive speed selection
        speed_name, delay, frame_skip = get_speed_mode()
        
        # Performance tracking
        total_inference_time = 0
        total_steps = 0
        
        # Test the model
        for episode in range(5):  # Test 5 episodes
            state, _ = env.reset()
            state = preprocess_state(state)
            
            episode_reward = 0
            steps = 0
            
            print(f"\n🎮 Episode {episode + 1} - Speed: {speed_name.upper()}")
            
            # Initialize frame stack with the first frame
            frame_stack = np.stack([state] * 4, axis=0)
            
            # Pre-convert to tensor for efficiency (reuse same tensor)
            if torch.cuda.is_available():
                frame_tensor = torch.cuda.FloatTensor(1, 4, 84, 84)
            else:
                frame_tensor = torch.FloatTensor(1, 4, 84, 84)
            
            while True:
                # Efficient tensor copying (reuse pre-allocated tensor)
                frame_tensor[0] = torch.from_numpy(frame_stack)
                
                # FAST inference - no gradient computation needed
                action_probs, value = model(frame_tensor)
                
                # Ultra-fast action selection (no .item() calls in loop)
                action = torch.argmax(action_probs, dim=1)[0]
                action_int = int(action)  # Convert once
                
                # Print action info less frequently for speed
                if steps % 50 == 0 and speed_mode != "ultra":
                    action_names = ["NOOP", "Fire", "Up", "Right", "Left", "Down", 
                                  "Up-Right", "Up-Left", "Down-Right", "Down-Left",
                                  "Up-Fire", "Right-Fire", "Left-Fire", "Down-Fire",
                                  "Up-Right-Fire", "Up-Left-Fire", "Down-Right-Fire", "Down-Left-Fire"]
                    action_name = action_names[action_int] if action_int < len(action_names) else f"Action-{action_int}"
                    confidence = float(torch.max(action_probs))
                    print(f"  Step {steps:3d}: Action {action_int:2d} ({action_name}) - Confidence: {confidence:.2f}")
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                
                # Efficient frame stack update (in-place operations)
                next_state = preprocess_state(next_state)
                frame_stack[:-1] = frame_stack[1:]  # Shift frames efficiently
                frame_stack[-1] = next_state  # Add new frame
                
                # Dynamic speed control
                if speed_mode == "ultra":
                    pass  # No delay for maximum speed
                elif steps % 10 == 0:  # Only delay every 10th step for speed
                    time.sleep(speed_delays[speed_mode])
                
                if reward != 0:
                    print(f"    💰 Reward: {reward}, Total: {episode_reward}")
                
                if done or steps > 1000:  # Limit episode length
                    print(f"  Episode {episode + 1} finished: {episode_reward} points in {steps} steps")
                    break
        
        env.close()
        print("🏁 Testing completed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_and_test_model()
