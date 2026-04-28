import os
import numpy as np
import matplotlib.pyplot as plt
from utils.logger import get_logger
from utils.config import cfg

log = get_logger(__name__)

# Baseline reference values (new phase-aware environment)
RANDOM_BASELINE    = cfg["evaluation"].get("random_baseline", 1443)
FIFO_BASELINE      = cfg["evaluation"].get("fifo_baseline", 3872)
THRESHOLD_BASELINE = cfg["evaluation"].get("threshold_baseline", -502)


def moving_average(values, window=50):
    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        averages.append(np.mean(values[start:i + 1]))
    return np.array(averages)


def plot_single():
    """Plot a single training run from results/rewards.npy"""
    rewards_path = "results/rewards.npy"
    if not os.path.exists(rewards_path):
        log.warning("Reward file not found: %s", rewards_path)
        return

    rewards = np.load(rewards_path)
    window  = cfg["evaluation"]["plot_window"]
    avg     = moving_average(rewards, window=window)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="raw reward", alpha=0.35)
    plt.plot(avg, label=f"DDQN (avg-{window})", linewidth=2)
    plt.axhline(y=RANDOM_BASELINE,    color="red",    linestyle="--", label=f"random baseline ({RANDOM_BASELINE})")
    plt.axhline(y=FIFO_BASELINE,      color="green",  linestyle="--", label=f"FIFO baseline ({FIFO_BASELINE})")
    plt.axhline(y=THRESHOLD_BASELINE, color="orange", linestyle="--", label=f"threshold baseline ({THRESHOLD_BASELINE})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("ProScale DDQN Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = "results/reward_curve.png"
    plt.savefig(output_path)
    log.info("Saved single-seed plot to %s", output_path)
    plt.show()


def plot_multi_seed(seeds=None):
    """Plot mean ± std across multiple seeds."""
    seeds       = seeds or cfg["evaluation"]["seeds"]
    window      = cfg["evaluation"]["plot_window"]
    all_rewards = []

    for seed in seeds:
        path = f"results/rewards_seed{seed}.npy"
        if os.path.exists(path):
            rewards = np.load(path)
            all_rewards.append(moving_average(rewards, window=window))
        else:
            log.warning("Seed file not found, skipping: %s", path)

    if not all_rewards:
        log.error("No seed files found — run training with seeds first.")
        return

    min_len     = min(len(r) for r in all_rewards)
    all_rewards = np.array([r[:min_len] for r in all_rewards])
    mean        = np.mean(all_rewards, axis=0)
    std         = np.std(all_rewards, axis=0)
    episodes    = np.arange(min_len)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, mean, color="steelblue", linewidth=2,
             label=f"mean reward (n={len(all_rewards)} seeds)")
    plt.fill_between(episodes, mean - std, mean + std,
                     alpha=0.25, color="steelblue", label="± 1 std")
    plt.axhline(y=RANDOM_BASELINE,    color="red",    linestyle="--", linewidth=1.5,
                label=f"random baseline ({RANDOM_BASELINE})")
    plt.axhline(y=FIFO_BASELINE,      color="green",  linestyle="--", linewidth=1.5,
                label=f"FIFO baseline ({FIFO_BASELINE})")
    plt.axhline(y=THRESHOLD_BASELINE, color="orange", linestyle="--", linewidth=1.5,
                label=f"threshold baseline ({THRESHOLD_BASELINE})")
    plt.xlabel("Episode")
    plt.ylabel(f"Reward (moving avg, window={window})")
    plt.title(f"ProScale DDQN Training Reward Curve — Mean ± Std over {len(all_rewards)} Seeds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = "results/reward_curve_multiseed.png"
    plt.savefig(output_path, dpi=150)
    log.info("Saved multi-seed plot to %s", output_path)
    log.info("Final reward | mean=%.1f | std=%.1f", mean[-1], std[-1])
    plt.show()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "multi":
        plot_multi_seed()
    else:
        plot_single()
