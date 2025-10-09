# test_mario_model_fast.py
import os
import time
import math
import cv2
import numpy as np
import torch
import gymnasium as gym
import ale_py

from train import CNNPolicy, make_env  # your existing modules (policy + atari wrappers)

# --- Optional: reduce overhead ---
os.environ["SDL_AUDIODRIVER"] = "dummy"       # avoid audio init
os.environ["SDL_RENDER_VSYNC"] = "0"          # if a SDL viewer sneaks in, prevent vsync
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(2)  # inference often faster with fewer CPU threads

# Register ALE envs (gymnasium>=1.0)
gym.register_envs(ale_py)

ACTION_NAMES = [
    "NOOP","Fire","Up","Right","Left","Down",
    "Up-Right","Up-Left","Down-Right","Down-Left",
    "Up-Fire","Right-Fire","Left-Fire","Down-Fire",
    "Up-Right-Fire","Up-Left-Fire","Down-Right-Fire","Down-Left-Fire"
]

def choose_speed():
    print("\n⚡ Choose playback mode:")
    print("1) SLOW        (delay=50 ms)")
    print("2) FAST        (delay=10 ms)")
    print("3) ULTRA       (delay=1 ms)")
    print("4) TURBO x3    (frame_skip=3)")
    print("5) HYPER x5    (frame_skip=5)")
    print("6) LUDICROUS x10 (frame_skip=10)")
    print("7) BENCHMARK   (no window, max speed)")
    choice = (input("Enter 1–7 [default 4]: ").strip() or "4")

    # delay is in milliseconds for cv2.waitKey
    presets = {
        "1": dict(delay_ms=50, frame_skip=1, render=True),
        "2": dict(delay_ms=10, frame_skip=1, render=True),
        "3": dict(delay_ms=1,  frame_skip=1, render=True),
        "4": dict(delay_ms=1,  frame_skip=3, render=True),
        "5": dict(delay_ms=1,  frame_skip=5, render=True),
        "6": dict(delay_ms=1,  frame_skip=10, render=True),
        "7": dict(delay_ms=0,  frame_skip=1, render=False),
    }
    return presets.get(choice, presets["4"])

def load_model(device):
    model = CNNPolicy(4, 18).to(device)
    # Try your latest checkpoint first; falls back to others automatically
    candidates = [
        "mario_ppo_checkpoint_20.pth",   # Try earlier checkpoint
        "mario_ppo_checkpoint_40.pth",
        "mario_ppo_checkpoint_60.pth",
        "mario_ppo_checkpoint_80.pth",
        "mario_ppo_checkpoint_100.pth",
        "mario_ppo_checkpoint_140.pth",
        # "mario_ppo_final.pth",
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        print(f"✅ Loaded model: {path}")
        model.eval()
        return model
    raise FileNotFoundError("No .pth checkpoint found. Train first.")

def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    else:
        print("💻 CPU inference")

    # Load policy
    model = load_model(device)

    # Speed config
    speed = choose_speed()
    delay_ms = speed["delay_ms"]
    frame_skip = speed["frame_skip"]
    use_render = speed["render"]

    # Simplified: Use single environment with proper rendering mode
    if use_render:
        # Use human rendering mode for direct visualization
        env = gym.make("ALE/MarioBros-v5", render_mode="human")
        # Apply same preprocessing as training
        from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
        from gymnasium.wrappers import FrameStackObservation
        env = AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=True)
        env = FrameStackObservation(env, stack_size=4)
        print("� Visual mode with direct rendering")
    else:
        # Pure speed benchmark (no rendering)
        env = make_env("ALE/MarioBros-v5", stack=4)
        print("🔥 Benchmark mode (no rendering)")

    # Single environment reset
    env.reset()

    total_steps, total_infer_s, action_hist = 0, 0.0, np.zeros(18, dtype=np.int64)

    episodes = 3
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_ret, steps, last_action = 0.0, 0, 0
        
        print(f"\n🎮 Episode {ep+1} starting...")
        
        while True:
            # Frame skip: repeat last action without running policy
            if frame_skip > 1 and (steps % frame_skip != 0):
                a = last_action
            else:
                t0 = time.perf_counter()
                x = torch.from_numpy(np.array(obs)).unsqueeze(0).to(device)  # [1,4,84,84]
                with torch.no_grad():
                    logits, _ = model(x)
                    # Apply NOOP bias to encourage movement (same as training)
                    logits = logits.clone()
                    logits[:, 0] -= math.log(10.0)  # Reduce NOOP probability
                    
                    # Add temperature for more diverse actions
                    temperature = 2.0  # Higher = more exploration
                    logits_temp = logits / temperature
                    a = int(torch.distributions.Categorical(logits=logits_temp).sample().item())
                    
                    # Debug action selection occasionally
                    if steps % 100 == 0:
                        action_probs = torch.softmax(logits, dim=-1)
                        top_actions = torch.topk(action_probs, 3)
                        print(f"  Step {steps}: Action {a} ({ACTION_NAMES[a]}) - "
                              f"Top probs: {[f'{ACTION_NAMES[idx]}:{val:.3f}' for idx, val in zip(top_actions.indices[0], top_actions.values[0])]}")
                        
                total_infer_s += (time.perf_counter() - t0)
                last_action = a

            # Step environment 
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_ret += reward
            action_hist[a] += 1
            steps += 1
            total_steps += 1

            # Add small delay for visual modes
            if use_render and delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            obs = next_obs

            # Print rewards when they occur
            if reward != 0:
                print(f"    💰 Reward: {reward} at step {steps} (total: {ep_ret})")

            # safety cap for demos
            if done or steps > 3000:
                print(f"Episode {ep+1}: return={ep_ret:.1f}, steps={steps}")
                break

    env.close()

    # Perf stats
    if total_steps > 0 and total_infer_s > 0:
        fps = total_steps / total_infer_s
        eff_fps = fps * frame_skip
        print(f"\n🏁 Done. Inference FPS: {fps:.1f} | Effective FPS (skip x{frame_skip}): {eff_fps:.1f}")
    print("\nTop actions:", ", ".join(
        f"{ACTION_NAMES[i]}:{c}" for i, c in list(sorted(enumerate(action_hist), key=lambda x: -x[1]))[:5]
    ))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        raise
