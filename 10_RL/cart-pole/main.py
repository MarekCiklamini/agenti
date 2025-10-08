import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value


def gae_advantages(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards, values, dones: [T, N]
    returns adv, returns: [T, N]
    """
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    lastgaelam = np.zeros(N, dtype=np.float32)
    # append bootstrap value = 0 for terminal
    values_ext = np.concatenate([values, np.zeros((1, N), dtype=np.float32)], axis=0)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values_ext[t + 1] * (1.0 - dones[t]) - values[t]
        lastgaelam = delta + gamma * lam * (1.0 - dones[t]) * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns


def make_envs(env_id, num_envs, seed):
    env = gym.vector.make(env_id, num_envs=num_envs, asynchronous=False)
    env.reset(seed=seed)
    return env


def ppo_update(
    model, optimizer, obs, actions, old_logprobs, advantages, returns, clip_coef=0.2,
    vf_coef=0.5, ent_coef=0.01, epochs=4, minibatch_size=1024, max_grad_norm=0.5, device="cpu"
):
    B = obs.shape[0]
    inds = np.arange(B)
    for _ in range(epochs):
        np.random.shuffle(inds)
        for start in range(0, B, minibatch_size):
            mb = inds[start:start + minibatch_size]
            mb_obs = torch.as_tensor(obs[mb], dtype=torch.float32, device=device)
            mb_actions = torch.as_tensor(actions[mb], dtype=torch.long, device=device)
            mb_oldlog = torch.as_tensor(old_logprobs[mb], dtype=torch.float32, device=device)
            mb_adv = torch.as_tensor(advantages[mb], dtype=torch.float32, device=device)
            mb_ret = torch.as_tensor(returns[mb], dtype=torch.float32, device=device)

            logits, value = model(mb_obs)
            dist = Categorical(logits=logits)
            new_logprob = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = (new_logprob - mb_oldlog).exp()
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)

            # Policy loss
            pg1 = ratio * mb_adv
            pg2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
            pg_loss = -torch.min(pg1, pg2).mean()

            # Value loss (clipped)
            with torch.no_grad():
                v_clipped = value + (torch.clamp(value - mb_ret, -clip_coef, clip_coef))
            v_loss_unclipped = (value - mb_ret).pow(2)
            v_loss_clipped = (v_clipped - mb_ret).pow(2)
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval", action="store_true", help="Run a rendered eval episode after training")
    args = parser.parse_args()

    device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Envs
    env = make_envs(args.env_id, args.num_envs, args.seed)

    obs_space = env.single_observation_space
    act_space = env.single_action_space
    obs_dim = obs_space.shape[0]
    n_actions = act_space.n

    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Storage
    T = args.rollout_len
    N = args.num_envs
    obs, _ = env.reset(seed=args.seed)
    episode_returns = np.zeros(N, dtype=np.float32)
    episode_lengths = np.zeros(N, dtype=np.int32)
    completed_returns = []

    steps_done = 0
    while steps_done < args.total_steps:
        buf_obs = np.zeros((T, N, obs_dim), dtype=np.float32)
        buf_actions = np.zeros((T, N), dtype=np.int64)
        buf_logprobs = np.zeros((T, N), dtype=np.float32)
        buf_rewards = np.zeros((T, N), dtype=np.float32)
        buf_dones = np.zeros((T, N), dtype=np.float32)
        buf_values = np.zeros((T, N), dtype=np.float32)

        for t in range(T):
            buf_obs[t] = obs

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action, logp, entropy, value = model.act(obs_t)
            actions = action.cpu().numpy()
            logprobs = logp.cpu().numpy()
            values = value.cpu().numpy()

            buf_actions[t] = actions
            buf_logprobs[t] = logprobs
            buf_values[t] = values

            obs, reward, terminated, truncated, info = env.step(actions)
            done = np.logical_or(terminated, truncated).astype(np.float32)

            buf_rewards[t] = reward
            buf_dones[t] = done

            # Track episodic return for logging
            episode_returns += reward
            episode_lengths += 1
            for i, d in enumerate(done):
                if d:
                    completed_returns.append(episode_returns[i])
                    episode_returns[i] = 0.0
                    episode_lengths[i] = 0

            steps_done += N

        # Compute advantages
        adv, ret = gae_advantages(
            rewards=buf_rewards, values=buf_values, dones=buf_dones, gamma=0.99, lam=0.95
        )

        # Flatten to [T*N, ...]
        B = T * N
        flat_obs = buf_obs.reshape(B, obs_dim)
        flat_actions = buf_actions.reshape(B)
        flat_logprobs = buf_logprobs.reshape(B)
        flat_adv = adv.reshape(B)
        flat_ret = ret.reshape(B)

        ppo_update(
            model, optimizer,
            obs=flat_obs, actions=flat_actions, old_logprobs=flat_logprobs,
            advantages=flat_adv, returns=flat_ret,
            clip_coef=0.2, vf_coef=0.5, ent_coef=0.01,
            epochs=args.epochs, minibatch_size=args.minibatch_size, device=device
        )

        if len(completed_returns) >= 10:
            print(f"[steps={steps_done}] avg_return(last10) = {np.mean(completed_returns[-10:]):.1f}")

    # Optional render eval
    if args.eval:
        eval_env = gym.make(args.env_id, render_mode="human")
        obs, _ = eval_env.reset(seed=args.seed + 1)
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                logits, _ = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                action = torch.argmax(logits, dim=-1).item()
            obs, r, term, trunc, _ = eval_env.step(action)
            total += r
            done = term or trunc
        print(f"Eval return: {total:.1f}")


if __name__ == "__main__":
    main()
