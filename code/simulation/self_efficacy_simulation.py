"""Two-action TD simulation with the original Zorowitz efficacy backup.

The simulated environment is a bidirectional 10-state chain.  Interior states
have two actions (left and right), so the maximum and minimum successor-action
values are genuinely distinct.  Arrival at the left endpoint terminates with
reward 0; arrival at the right endpoint terminates with Bernoulli reward.

For a nonterminal successor state, the TD target uses Eq. 3 of Zorowitz,
Momennejad, and Daw (2020):

    r + gamma * [w * max_a Q(s', a) + (1 - w) * min_a Q(s', a)]

The original model constrains 0 <= w <= 1.  The agent's action policy is still
maximizing (implemented as epsilon-greedy during learning so both actions can
be sampled), and the value used for plotting is therefore max_a Q_w(s, a).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


class LinearTrack:
    """Bidirectional linear chain with nonrewarded and rewarded endpoints."""

    LEFT = 0
    RIGHT = 1
    ACTION_NAMES = ("left", "right")

    def __init__(self, n_states: int = 10, reward_prob: float = 0.85):
        if n_states < 3:
            raise ValueError("n_states must include at least one interior state")
        if not 0.0 <= reward_prob <= 1.0:
            raise ValueError("reward_prob must lie in [0, 1]")
        self.n_states = int(n_states)
        self.left_terminal = 0
        self.right_terminal = self.n_states - 1
        self.reward_prob = float(reward_prob)
        self.current_state = 1

    def reset(self, rng: np.random.RandomState) -> int:
        """Start uniformly at one of the interior states."""
        self.current_state = int(rng.randint(1, self.right_terminal))
        return self.current_state

    def step(
        self,
        action: int,
        rng: np.random.RandomState,
    ) -> Tuple[int, float, bool]:
        """Move one state left/right and terminate on arrival at an endpoint."""
        if action == self.LEFT:
            next_state = self.current_state - 1
        elif action == self.RIGHT:
            next_state = self.current_state + 1
        else:
            raise ValueError(f"Unknown action: {action}")

        self.current_state = next_state
        if next_state == self.left_terminal:
            return next_state, 0.0, True
        if next_state == self.right_terminal:
            reward = float(rng.random_sample() < self.reward_prob)
            return next_state, reward, True
        return next_state, 0.0, False


class ValueLearner:
    """Tabular TD learner using the Zorowitz weighted-extrema target."""

    def __init__(
        self,
        n_states: int,
        alpha: float = 0.1,
        gamma: float = 0.8,
        w: float = 1.0,
    ):
        if not 0.0 <= w <= 1.0:
            raise ValueError(
                "The original Zorowitz efficacy weight w must lie in [0, 1]."
            )
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")

        self.n_states = int(n_states)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.w = float(w)
        self.Q = np.zeros((self.n_states, 2), dtype=float)

    def weighted_extrema(self, action_values: np.ndarray) -> float:
        """Return w*max(Q) + (1-w)*min(Q) for one successor state."""
        return float(
            self.w * np.max(action_values)
            + (1.0 - self.w) * np.min(action_values)
        )

    def choose_action(
        self,
        state: int,
        rng: np.random.RandomState,
        epsilon: float,
    ) -> int:
        """Choose epsilon-greedily, breaking ties randomly and reproducibly."""
        if rng.random_sample() < epsilon:
            return int(rng.randint(0, 2))
        best_actions = np.flatnonzero(self.Q[state] == np.max(self.Q[state]))
        return int(best_actions[rng.randint(0, len(best_actions))])

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Apply one TD update and return the prediction error."""
        future = 0.0 if done else self.weighted_extrema(self.Q[next_state])
        target = reward + self.gamma * future
        delta = target - self.Q[state, action]
        self.Q[state, action] += self.alpha * delta
        return float(delta)

    def greedy_state_value(self) -> np.ndarray:
        """Return max_a Q_w(s,a), consistent with the model's choice rule."""
        return np.max(self.Q, axis=1).copy()


def train_agent(
    env: LinearTrack,
    agent: ValueLearner,
    rng: np.random.RandomState,
    n_episodes: int = 100,
    epsilon: float = 0.20,
    max_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Train one agent and return episode rewards, lengths, and timeouts."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")

    episode_rewards = np.zeros(n_episodes, dtype=float)
    episode_lengths = np.zeros(n_episodes, dtype=int)
    timeout_count = 0

    for episode in range(n_episodes):
        state = env.reset(rng)
        total_reward = 0.0

        for step_index in range(max_steps):
            action = agent.choose_action(state, rng, epsilon)
            next_state, reward, done = env.step(action, rng)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            if done:
                episode_lengths[episode] = step_index + 1
                break
        else:
            timeout_count += 1
            episode_lengths[episode] = max_steps

        episode_rewards[episode] = total_reward

    return episode_rewards, episode_lengths, timeout_count


def run_simulation(
    w_values: Sequence[float],
    n_runs: int = 20,
    n_episodes: int = 100,
    n_states: int = 10,
    alpha: float = 0.1,
    gamma: float = 0.8,
    reward_prob: float = 0.85,
    epsilon: float = 0.20,
    max_steps: int = 200,
    seed: int = 1,
) -> Dict:
    """Run the two-action simulation for each efficacy weight.

    A single legacy ``RandomState`` stream is initialized once, then consumed
    with efficacy weight as the outer loop and run as the inner loop.  This
    explicit order makes the published result exactly reproducible.
    """
    w_values = np.asarray(w_values, dtype=float)
    if np.any((w_values < 0.0) | (w_values > 1.0)):
        raise ValueError("All w values must lie in the original [0, 1] domain")

    rng = np.random.RandomState(seed)
    positions = np.linspace(0.0, 90.0, n_states)
    value_functions = np.zeros((len(w_values), n_runs, n_states), dtype=float)
    action_values = np.zeros((len(w_values), n_runs, n_states, 2), dtype=float)
    episode_rewards = np.zeros((len(w_values), n_runs, n_episodes), dtype=float)
    episode_lengths = np.zeros((len(w_values), n_runs, n_episodes), dtype=int)
    timeout_counts = np.zeros((len(w_values), n_runs), dtype=int)

    for w_index, w in enumerate(w_values):
        print(f"Running simulations for w = {w:.2f}...")
        for run_index in range(n_runs):
            env = LinearTrack(n_states=n_states, reward_prob=reward_prob)
            agent = ValueLearner(
                n_states=n_states,
                alpha=alpha,
                gamma=gamma,
                w=float(w),
            )
            rewards, lengths, timeouts = train_agent(
                env,
                agent,
                rng,
                n_episodes=n_episodes,
                epsilon=epsilon,
                max_steps=max_steps,
            )

            values = agent.greedy_state_value()
            # Terminal Q rows are never updated because episodes end on
            # arrival.  For visualization/choice mapping, anchor them to the
            # known expected endpoint outcomes.
            values[env.left_terminal] = 0.0
            values[env.right_terminal] = reward_prob

            value_functions[w_index, run_index] = values
            action_values[w_index, run_index] = agent.Q
            episode_rewards[w_index, run_index] = rewards
            episode_lengths[w_index, run_index] = lengths
            timeout_counts[w_index, run_index] = timeouts

    return {
        "w_values": w_values,
        "value_functions": value_functions,
        "action_values": action_values,
        "positions": positions,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "timeout_counts": timeout_counts,
        "seed": int(seed),
        "backup_operator": "w*max(Q_next) + (1-w)*min(Q_next)",
        "reported_state_value": "max_a Q_w(s,a)",
        "action_names": LinearTrack.ACTION_NAMES,
        "endpoint_values": np.asarray([0.0, reward_prob]),
        "parameters": {
            "n_runs": int(n_runs),
            "n_episodes": int(n_episodes),
            "n_states": int(n_states),
            "alpha": float(alpha),
            "gamma": float(gamma),
            "reward_prob": float(reward_prob),
            "epsilon": float(epsilon),
            "max_steps": int(max_steps),
        },
    }


if __name__ == "__main__":
    demo = run_simulation([0.0, 0.5, 1.0], n_runs=2, n_episodes=20)
    for index, weight in enumerate(demo["w_values"]):
        mean_value = demo["value_functions"][index].mean(axis=0)
        print(f"w={weight:.2f}: {mean_value.round(3)}")
