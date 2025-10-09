# test2.py
# PPO for ALE/MarioBros-v5 with proper Atari preprocessing, GAE(λ), minibatches,
# entropy bonus, consistent NOOP biasing in logits, and stable training plumbing.
#
# Requirements:
#   pip install gymnasium ale-py torch numpy opencv-python python-dotenv
# Optional (for speed): pip install torch --index-url https://download.pytorch.org/whl/cu121
#
# Run:
#   python test2.py

import os
import math
import time
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
from gymnasium.wrappers import TransformObservation, FrameStackObservation
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing

# Register ALE environments
import ale_py
gym.register_envs(ale_py)

# --------------------
# Config
# --------------------
CONFIG = {
    "env_id": "ALE/MarioBros-v5",
    "total_iterations": 1000,        # PPO update iterations
    "steps_per_iter": 2048,          # env steps / iteration
    "num_stack": 4,                  # frame stack depth
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.1,                 # PPO clipping
    "epochs": 4,                     # PPO epochs per update
    "minibatch_size": 256,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "save_every": 20,
    "seed": 42,
    # Exploration nudging: apply a constant logit penalty to NOOP (action 0)
    "enable_noop_bias": True,
    "noop_bias_factor": 10.0,        # logits[:,0] -= log(noop_bias_factor)
    # Logging
    "print_every": 10,
}

# --------------------
# Torch perf toggles
# --------------------
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if hasattr(torch.backends, "opt_einsum"):
    torch.backends.opt_einsum.enabled = True
torch.set_float32_matmul_precision("medium")
torch.set_num_threads(6)

# --------------------
# Utils
# --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_env(env_id: str, stack: int = 4):
    """
    Standard Atari preprocessing:
    - grayscale, resize to 84x84
    - frame_skip already built into ALE/MarioBros-v5 (frameskip=4)
    - no scaling inside AtariPreprocessing; we'll scale once via TransformObservation
    - FrameStack for temporal info
    """
    env = gym.make(env_id, render_mode=None)
    # Don't use frame_skip since ALE/MarioBros-v5 already has frameskip=4
    # Use scale_obs=True to normalize observations to [0,1] range
    env = AtariPreprocessing(env, frame_skip=1, screen_size=84, grayscale_obs=True, scale_obs=True)
    env = FrameStackObservation(env, stack_size=stack)
    return env

# --------------------
# Networks
# --------------------
class CNNPolicy(nn.Module):
    """
    CNN torso + dual heads (policy logits, value).
    Assumes input shape: [B, C=stack, 84, 84] with float32 in [0,1].
    """
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        # compute conv out
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            conv_out = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(conv_out, 512), nn.ReLU())
        self.pi = nn.Linear(512, num_actions)  # logits
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value

# --------------------
# PPO Agent
# --------------------
class PPOAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Device: {self.device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)} | CUDA: {torch.version.cuda}")

        self.obs_shape = self.env.observation_space.shape  # (stack, 84, 84)
        self.num_actions = self.env.action_space.n

        self.net = CNNPolicy(self.obs_shape[0], self.num_actions).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg["learning_rate"])

        # rollout buffers
        self.reset_storage()

        # episode tracking
        self.ep_return = 0.0
        self.ep_returns = []

    def reset_storage(self):
        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.done_buf = []

    @torch.no_grad()
    def choose_action(self, obs_t):
        logits, value = self.net(obs_t)

        # Apply a soft NOOP penalty only at logit level
        if self.cfg.get("enable_noop_bias", True):
            logits = logits.clone()
            logits[:, 0] -= math.log(self.cfg.get("noop_bias_factor", 5.0))  # reduce NOOP likelihood

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


    def collect_rollout(self, steps):
        obs, _ = self.env.reset()
        for _ in range(steps):
            obs_t = torch.from_numpy(np.array(obs)).unsqueeze(0).to(self.device)  # [1,C,84,84]
            action, logp, value = self.choose_action(obs_t)

            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated

            self.obs_buf.append(obs_t.squeeze(0).cpu().numpy())
            self.act_buf.append(action.item())
            self.logp_buf.append(logp.item())
            self.rew_buf.append(float(reward))
            self.val_buf.append(value.item())
            self.done_buf.append(float(done))

            # episode tracking
            self.ep_return += float(reward)
            if done:
                self.ep_returns.append(self.ep_return)
                self.ep_return = 0.0
                next_obs, _ = self.env.reset()

            obs = next_obs

    def compute_gae(self, rewards, values, dones, gamma, lam):
        """
        rewards: [T]
        values:  [T] (no bootstrap value at T)
        dones:   [T] (1.0 if done)
        returns: [T]
        adv:     [T]
        """
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        ret = np.zeros(T, dtype=np.float32)
        next_value = 0.0
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
            ret[t] = adv[t] + values[t]
            next_value = values[t]
        return ret, adv

    def ppo_update(self):
        cfg = self.cfg

        # tensors
        obs = torch.from_numpy(np.array(self.obs_buf)).to(self.device)         # [N, C, 84, 84]
        acts = torch.tensor(self.act_buf, dtype=torch.long, device=self.device)
        old_logp = torch.tensor(self.logp_buf, dtype=torch.float32, device=self.device)
        rews = np.array(self.rew_buf, dtype=np.float32)
        vals = np.array(self.val_buf, dtype=np.float32)
        dones = np.array(self.done_buf, dtype=np.float32)

        # returns / advantages
        ret, adv = self.compute_gae(rews, vals, dones, cfg["gamma"], cfg["gae_lambda"])
        rets = torch.tensor(ret, dtype=torch.float32, device=self.device)
        advs = torch.tensor(adv, dtype=torch.float32, device=self.device)
        # normalize advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # training loop with minibatches
        num_samples = obs.shape[0]
        idxs = np.arange(num_samples)

        policy_loss_hist, value_loss_hist, entropy_hist = [], [], []

        for _ in range(cfg["epochs"]):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, cfg["minibatch_size"]):
                mb = idxs[start:start + cfg["minibatch_size"]]
                mb_obs = obs[mb]
                mb_acts = acts[mb]
                mb_old_logp = old_logp[mb]
                mb_advs = advs[mb]
                mb_rets = rets[mb]

                logits, values = self.net(mb_obs)
                # apply the SAME NOOP biasing rule used during sampling
                if cfg["enable_noop_bias"]:
                    logits = logits.clone()
                    # logits[:, 0] -= math.log(cfg["noop_bias_factor"])
                    logits[:, 0] -= math.log(self.cfg.get("noop_bias_factor", 5.0))
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(mb_acts)

                entropy = dist.entropy().mean()

                # PPO objective
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - cfg["clip_eps"], 1.0 + cfg["clip_eps"]) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, mb_rets)

                loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg["max_grad_norm"])
                self.opt.step()

                policy_loss_hist.append(policy_loss.item())
                value_loss_hist.append(value_loss.item())
                entropy_hist.append(entropy.item())

        # clear buffers
        self.reset_storage()

        return (
            float(np.mean(policy_loss_hist)),
            float(np.mean(value_loss_hist)),
            float(np.mean(entropy_hist)),
        )

    def save(self, path):
        state = {
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": self.cfg,
        }
        torch.save(state, path)

# --------------------
# Main
# --------------------
def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    print(f"🎮 Creating environment: {cfg['env_id']}")
    env = make_env(cfg["env_id"], stack=cfg["num_stack"])

    agent = PPOAgent(env, cfg)
    print(f"Action space: {env.action_space.n} | Obs shape: {env.observation_space.shape}")
    print("Action mapping (ALE standard):")
    print("0: NOOP, 1: Fire, 2: Up, 3: Right, 4: Left, 5: Down")
    print("6: Up-Right, 7: Up-Left, 8: Down-Right, 9: Down-Left")
    print("10: Up-Fire, 11: Right-Fire, 12: Left-Fire, 13: Down-Fire")
    print("14: Up-Right-Fire, 15: Up-Left-Fire, 16: Down-Right-Fire, 17: Down-Left-Fire")

    start_time = time.time()
    for it in range(cfg["total_iterations"]):
        agent.collect_rollout(cfg["steps_per_iter"])
        pol_loss, val_loss, ent = agent.ppo_update()

        if (it % cfg["print_every"]) == 0:
            avg_ret = np.mean(agent.ep_returns[-10:]) if agent.ep_returns else 0.0
            print(
                f"[Iter {it:04d}] "
                f"AvgEpRet(10): {avg_ret:7.2f} | "
                f"Policy: {pol_loss:7.4f} | Value: {val_loss:7.4f} | Entropy: {ent:6.4f} | "
                f"Elapsed: {time.time()-start_time:7.1f}s"
            )

        if (it % cfg["save_every"]) == 0:
            path = f"mario_ppo_checkpoint_{it}.pth"
            agent.save(path)
            print(f"💾 Saved: {path}")

    # final save
    agent.save("mario_ppo_final.pth")
    print("🏁 Training complete. Saved mario_ppo_final.pth")
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
