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
        "mario_ppo_checkpoint_140.pth",
        # "mario_ppo_final.pth",
        # "mario_ppo_model_100.pth",
        # "mario_ppo_model_80.pth",
        # "mario_ppo_model_60.pth",
        # "mario_ppo_model_40.pth",
        # "mario_ppo_model_20.pth",
        # "mario_ppo_model_0.pth",
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

    # Use rgb_array for uncapped speed (we will draw with OpenCV)
    # Keep training-identical preprocessing via your make_env if not rendering:
    if use_render:
        env = gym.make("ALE/MarioBros-v5", render_mode="rgb_array")
        # Important: we still want the same obs preprocessing/stacking the policy expects.
        # Reuse training wrappers (AtariPreprocessing + FrameStack + scaling) by building a parallel obs-only env
        # Easiest: create a non-rendering env for observations, but step the same actions as the render env.
        obs_env = make_env("ALE/MarioBros-v5", stack=4)  # returns scaled float [0,1], shape [4,84,84]
        print("🎥 Display via OpenCV from rgb_array, policy obs from training wrappers.")
    else:
        # Pure speed benchmark (no rendering)
        env = make_env("ALE/MarioBros-v5", stack=4)
        obs_env = env  # same source

    # Reset both envs in sync
    obs_env.reset()
    if use_render:
        env.reset()

    total_steps, total_infer_s, action_hist = 0, 0.0, np.zeros(18, dtype=np.int64)

    if use_render:
        cv2.namedWindow("Mario", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Mario", 640, 480)

    episodes = 3
    for ep in range(episodes):
        obs, _ = obs_env.reset()
        if use_render:
            env.reset()

        ep_ret, steps, last_action = 0.0, 0, 0
        while True:
            # Frame skip: repeat last action without running policy
            if frame_skip > 1 and (steps % frame_skip != 0):
                a = last_action
            else:
                t0 = time.perf_counter()
                x = torch.from_numpy(np.array(obs)).unsqueeze(0).to(device)  # [1,4,84,84]
                with torch.no_grad():
                    logits, _ = model(x)
                    # Greedy for viewing stability; use sampling if desired:
                    # a = int(torch.distributions.Categorical(logits=logits).sample().item())
                    a = int(torch.argmax(logits, dim=-1).item())
                total_infer_s += (time.perf_counter() - t0)
                last_action = a

            # Step *both* envs to keep obs and rgb in sync
            next_obs, reward, terminated, truncated, _ = obs_env.step(a)
            done = terminated or truncated
            ep_ret += reward
            action_hist[a] += 1
            steps += 1
            total_steps += 1

            if use_render:
                frame = env.render()  # returns RGB array HxWx3 (uint8)
                # Fast resize for nicer viewing without extra work in env:
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Mario", frame)
                # waitKey is the only UI throttle; set to 1ms (or 0..n for extra delay)
                key = cv2.waitKey(delay_ms)
                if key == 27:  # ESC to quit early
                    done = True

            obs = next_obs

            # safety cap for demos
            if done or steps > 3000:
                print(f"Episode {ep+1}: return={ep_ret:.1f}, steps={steps}")
                break

    if use_render:
        cv2.destroyAllWindows()
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
