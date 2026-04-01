import os
import numpy as np
import matplotlib.pyplot as plt

def moving_average(values, window=10):
    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        averages.append(np.mean(values[start:i + 1]))
    return np.array(averages)

def main():
    rewards_path = "results/rewards.npy"

    if not os.path.exists(rewards_path):
        print(f"File not found: {rewards_path}")
        print("Please run training first to generate rewards.npy")
        return
    
    rewards = np.load(rewards_path)
    avg10 = moving_average(rewards, window=10)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="raw reward", alpha=0.35)
    plt.plot(avg10, label="moving average (10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DDQN Training Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = "results/reward_curve.png"
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()


