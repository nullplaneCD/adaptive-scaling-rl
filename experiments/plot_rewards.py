import os
import numpy as np
import matplotlib.pyplot as plt

RANDOM_BASELINE = 2654

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
        print(f"File not found: {rewards_path}")
        return

    rewards = np.load(rewards_path)
    avg = moving_average(rewards, window=50)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="raw reward", alpha=0.35)
    plt.plot(avg, label="moving average (50)")
    plt.axhline(y=RANDOM_BASELINE, color='red', linestyle='--', label=f"random baseline ({RANDOM_BASELINE})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDQN Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = "results/reward_curve.png"
    plt.savefig(output_path)
    print(f"Saved single plot to {output_path}")
    plt.show()

def plot_multi_seed(seeds=range(10)):
    """Plot mean ± std across multiple seeds"""
    all_rewards = []

    for seed in seeds:
        path = f"results/rewards_seed{seed}.npy"
        if os.path.exists(path):
            rewards = np.load(path)
            avg = moving_average(rewards, window=50)
            all_rewards.append(avg)
        else:
            print(f"Warning: {path} not found, skipping seed {seed}")

    if not all_rewards:
        print("No seed files found. Run training with seeds first.")
        return

    # Align lengths
    min_len = min(len(r) for r in all_rewards)
    all_rewards = np.array([r[:min_len] for r in all_rewards])

    mean = np.mean(all_rewards, axis=0)
    std = np.std(all_rewards, axis=0)
    episodes = np.arange(min_len)

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, mean, color='steelblue', linewidth=2, label=f"mean reward (n={len(all_rewards)} seeds)")
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.25, color='steelblue', label="± 1 std")
    plt.axhline(y=RANDOM_BASELINE, color='red', linestyle='--', linewidth=1.5, label=f"random baseline ({RANDOM_BASELINE})")
    plt.xlabel("Episode")
    plt.ylabel("Reward (moving avg, window=50)")
    plt.title(f"DDQN Training Reward Curve — Mean ± Std over {len(all_rewards)} Seeds")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = "results/reward_curve_multiseed.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved multi-seed plot to {output_path}")
    print(f"Final reward — Mean: {mean[-1]:.1f}, Std: {std[-1]:.1f}")
    plt.show()

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    if mode == "multi":
        plot_multi_seed()
    else:
        plot_single()
