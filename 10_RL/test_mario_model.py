# test_mario_model.py
import gymnasium as gym
import torch
import ale_py
import numpy as np
import cv2
import os
import time
from train import CNNPolicy, make_env

# Register ALE environments
gym.register_envs(ale_py)

# Performance optimization settings
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
torch.set_num_threads(6)  # Optimize CPU usage (increased for better performance)

# Additional PyTorch optimizations
if hasattr(torch.backends, 'opt_einsum'):
    torch.backends.opt_einsum.enabled = True
torch.set_float32_matmul_precision('medium')  # Use Tensor Cores if available

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
    """Interactive speed selection with extreme modes"""
    print("\n⚡ Choose playback speed:")
    print("1. 🐌 SLOW - Easy to watch (0.05s delay)")
    print("2. 🚀 FAST - Quick playback (0.01s delay)")  
    print("3. ⚡ ULTRA - Maximum speed (no delay)")
    print("4. 🎯 TURBO - Skip frames (3x faster)")
    print("5. 🏎️  HYPER - Skip more frames (5x faster)")
    print("6. 🚁 LUDICROUS - Maximum frame skip (10x faster)")
    print("7. 📊 BENCHMARK - No rendering, pure speed test")
    
    choice = input("Enter choice (1-7) [default: 4]: ").strip() or "4"
    
    modes = {
        "1": ("slow", 0.05, 1, True),
        "2": ("fast", 0.01, 1, True), 
        "3": ("ultra", 0.0, 1, True),
        "4": ("turbo", 0.0, 3, True),     # Skip 2/3 frames
        "5": ("hyper", 0.0, 5, True),     # Skip 4/5 frames  
        "6": ("ludicrous", 0.0, 10, True), # Skip 9/10 frames
        "7": ("benchmark", 0.0, 1, False)  # No rendering at all
    }
    
    return modes.get(choice, modes["4"])

def load_and_test_model():
    try:
        # Load the trained model with optimizations
        model = CNNPolicy(4, 18)  # 4 channels (stacked frames), 18 actions
        
        # Enable optimization modes for faster inference
        torch.set_grad_enabled(False)  # Disable gradient computation globally
        
        # GPU setup with detailed info
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("🚀 GPU ACCELERATION ENABLED!")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   Available Memory: {memory_gb:.1f} GB")
            
            # Clear GPU memory cache
            torch.cuda.empty_cache()
            model = model.cuda()
        else:
            print("💻 Using CPU inference")
        
        # Try to load different model checkpoints (load before compilation)
        model_files = [
            'mario_ppo_checkpoint_40.pth'
            # 'mario_ppo_final.pth',
            # 'mario_ppo_model_100.pth',
            # 'mario_ppo_model_80.pth', 
            # 'mario_ppo_model_60.pth',
            # 'mario_ppo_model_40.pth',
            # 'mario_ppo_model_20.pth',
            # 'mario_ppo_model_0.pth'
        ]
        
        model_loaded = False
        for model_file in model_files:
            try:
                if os.path.exists(model_file):
                    print(f"Loading model: {model_file}")
                    checkpoint = torch.load(model_file, map_location=device)
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'])
                    else:
                        model.load_state_dict(checkpoint)
                    model.eval()
                    model_loaded = True
                    
                    # Apply safe optimizations AFTER loading
                    print("⚡ Using optimized inference mode!")
                    
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
        
        print("🤖 Starting EXTREME SPEED Mario Bros Test!")
        print("⚡ Multiple speed modes available for ultimate performance!")
        print("🎯 Watch for lightning-fast decision making...")
        print("\nAction meanings:")
        print("0: NOOP, 1: Fire, 2: Up, 3: Right, 4: Left, 5: Down")
        print("6: Up-Right, 7: Up-Left, 8: Down-Right, 9: Down-Left")
        print("10+: Fire combinations")
        
        # Interactive speed selection
        speed_name, delay, frame_skip, use_rendering = get_speed_mode()
        
        # Create environment based on speed mode using the same setup as training
        if use_rendering:
            print(f"🎮 Creating visual environment for {speed_name.upper()} mode...")
            # Create base environment with rendering
            env = gym.make("ALE/MarioBros-v5", render_mode="human")
            # Apply the same preprocessing as training
            from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
            from gymnasium.wrappers import FrameStackObservation
            env = AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=True)
            env = FrameStackObservation(env, stack_size=4)
        else:
            print("🔥 BENCHMARK MODE: No rendering for maximum speed!")
            env = make_env("ALE/MarioBros-v5", stack=4)
        
        # Performance tracking
        total_inference_time = 0
        total_steps = 0
        total_actions = {i: 0 for i in range(18)}  # Track action distribution
        
        # Adjust episode count based on speed mode
        episode_count = 2 if speed_name in ["ludicrous", "benchmark"] else 5
        
        # Test the model
        for episode in range(episode_count):
            # Environment now returns properly stacked and preprocessed frames
            stacked_obs, _ = env.reset()
            
            episode_reward = 0
            steps = 0
            
            print(f"\n🎮 Episode {episode + 1} - Speed: {speed_name.upper()}")
            print(f"Observation shape: {stacked_obs.shape}, dtype: {stacked_obs.dtype}")
            
            # Cache last action for frame skipping
            last_action = 0
            
            while True:
                # Advanced frame skipping for extreme speeds
                if frame_skip > 1 and steps % frame_skip != 0:
                    # Reuse last action instead of NOOP for better continuity
                    action_int = last_action
                    
                    # Skip inference completely
                    stacked_obs, reward, terminated, truncated, _ = env.step(action_int)
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1
                    
                    if done or steps > 2000:  # Increased limit for speed tests
                        break
                    continue
                
                # Time the inference for performance measurement
                inference_start = time.perf_counter()
                
                # Convert to tensor (observations are already properly stacked and normalized)
                obs_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).to(device)
                
                # FASTEST inference - optimized forward pass
                with torch.no_grad():
                    logits, value = model(obs_tensor)
                    action_probs = torch.softmax(logits, dim=-1)
                
                # Lightning-fast action selection
                action_int = int(torch.argmax(action_probs))
                last_action = action_int  # Cache for frame skipping
                
                # Track inference time and actions
                inference_time = time.perf_counter() - inference_start
                total_inference_time += inference_time
                total_steps += 1
                total_actions[action_int] += 1
                
                # Minimal logging for extreme speeds
                log_frequency = {
                    "slow": 50, "fast": 100, "ultra": 200,
                    "turbo": 500, "hyper": 1000, "ludicrous": 2000, "benchmark": 5000
                }
                
                if steps % log_frequency.get(speed_name, 100) == 0 and speed_name not in ["ludicrous", "benchmark"]:
                    action_names = ["NOOP", "Fire", "Up", "Right", "Left", "Down", 
                                  "Up-Right", "Up-Left", "Down-Right", "Down-Left",
                                  "Up-Fire", "Right-Fire", "Left-Fire", "Down-Fire",
                                  "Up-Right-Fire", "Up-Left-Fire", "Down-Right-Fire", "Down-Left-Fire"]
                    action_name = action_names[action_int] if action_int < len(action_names) else f"Action-{action_int}"
                    confidence = float(torch.max(action_probs))
                    fps = 1.0 / inference_time if inference_time > 0 else float('inf')
                    print(f"  Step {steps:3d}: Action {action_int:2d} ({action_name}) - FPS: {fps:.0f}")
                
                # Take action in environment
                stacked_obs, reward, terminated, truncated, info = env.step(action_int)
                done = terminated or truncated
                
                episode_reward += reward
                steps += 1
                
                # Minimal delay control for extreme speeds
                if delay > 0 and steps % (20 if speed_name == "slow" else 50) == 0:
                    time.sleep(delay)
                
                # Minimal reward logging for speed
                if reward != 0 and speed_name in ["slow", "fast"]:
                    print(f"    💰 Reward: {reward}, Total: {episode_reward}")
                
                # Extended episode limits for speed tests
                episode_limit = {"benchmark": 5000, "ludicrous": 3000}.get(speed_name, 2000)
                if done or steps > episode_limit:
                    if speed_name not in ["benchmark"]:
                        print(f"  Episode {episode + 1} finished: {episode_reward} points in {steps} steps")
                    break
        
        env.close()
        
        # Advanced performance statistics
        avg_inference_time = total_inference_time / total_steps if total_steps > 0 else 0
        avg_fps = 1.0 / avg_inference_time if avg_inference_time > 0 else float('inf')
        
        # Calculate effective playback speed
        effective_fps = avg_fps * frame_skip if frame_skip > 1 else avg_fps
        
        print("\n🏁 Testing completed!")
        print(f"� EXTREME SPEED Performance Stats:")
        print(f"  Mode: {speed_name.upper()}")
        print(f"  Inference FPS: {avg_fps:.1f}")
        print(f"  Effective FPS: {effective_fps:.1f}")
        print(f"  Frame skip ratio: {frame_skip}x")
        print(f"  Average inference: {avg_inference_time*1000:.2f}ms")
        print(f"  Total inferences: {total_steps}")
        
        # Action distribution analysis
        most_used_actions = sorted(total_actions.items(), key=lambda x: x[1], reverse=True)[:5]
        action_names = ["NOOP", "Fire", "Up", "Right", "Left", "Down", 
                       "Up-Right", "Up-Left", "Down-Right", "Down-Left",
                       "Up-Fire", "Right-Fire", "Left-Fire", "Down-Fire",
                       "Up-Right-Fire", "Up-Left-Fire", "Down-Right-Fire", "Down-Left-Fire"]
        
        print(f"\n🎮 Action Analysis:")
        for action_id, count in most_used_actions:
            if count > 0:
                action_name = action_names[action_id] if action_id < len(action_names) else f"Action-{action_id}"
                percentage = (count / total_steps) * 100
                print(f"  {action_name}: {count} times ({percentage:.1f}%)")
        
        # Speed benchmarks
        if effective_fps > 1000:
            print("\n� LUDICROUS SPEED! Ultra-high performance achieved!")
        elif effective_fps > 500:
            print("\n🏎️  HYPER SPEED! Excellent optimization!")
        elif effective_fps > 200:
            print("\n🎯 TURBO SPEED! Great performance!")
        elif effective_fps > 60:
            print("\n🚀 HIGH SPEED! Very good performance!")
        else:
            print("\n⚠️  Consider using faster speed modes or GPU acceleration.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_and_test_model()
