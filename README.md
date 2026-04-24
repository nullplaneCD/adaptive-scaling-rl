# ProScale

**Proactive Cloud Resource Scaling via Reinforcement Learning**

> *ProScale learns your cloud workload rhythm and scales resources before demand arrives — not after.*

---

## Overview

ProScale is an intelligent resource scheduling engine that uses Deep Reinforcement Learning (Double DQN) to learn workload patterns and make proactive scaling decisions. Unlike conventional reactive auto-scalers such as Kubernetes HPA or AWS Auto Scaling — which respond to demand only after thresholds are breached — ProScale anticipates burst-quiet workload cycles and acts ahead of time.

**Key results (simulation, n=10 seeds):**

| Policy | Mean Episode Reward | vs ProScale |
|---|---|---|
| Kubernetes HPA-equivalent (threshold) | −502 | −110% |
| Random policy | 1,443 | −67% |
| FIFO scheduling | 3,872 | −12% |
| **ProScale (DDQN)** | **~4,400** | **—** |

ProScale consistently outperforms all baselines. Notably, the Kubernetes HPA-equivalent reactive policy performs **worse than random action selection** in patterned workload environments — a key empirical finding motivating this research.

---

## How It Works

ProScale models the resource scheduling problem as a Markov Decision Process (MDP) and trains a Double Deep Q-Network (DDQN) agent to learn an optimal policy through environment interaction.

### Phase-Aware Environment

Workload arrivals follow a **burst/quiet phase model** grounded in Google cluster trace observations (3× peak/off-peak ratio):

- **Burst phase**: arrival probability = 0.9 (33 steps)
- **Quiet phase**: arrival probability = 0.3 (33 steps)
- ~3 full cycles per 200-step episode

The agent observes both the current arrival probability and phase progress, enabling it to learn **anticipatory** (proactive) scaling — scaling up before a burst arrives, scaling down before the quiet phase wastes resources.

### Agent Architecture

```
Observation (26-dim) → [FC 256] → [ReLU] → [FC 256] → [ReLU] → Q-values (8 actions)
```

| Component | Specification |
|---|---|
| Algorithm | Double DQN (DDQN) |
| Network | 2-layer MLP, 256 hidden units |
| Optimiser | Adam (lr = 1e-4) |
| Loss | Huber loss (SmoothL1) |
| Replay buffer | 50,000 transitions |
| Epsilon schedule | 1.0 → 0.05, decay 0.999995 per step |
| Target update | Every 500 train steps |
| Warmup | 5,000 steps before training |

### Action Space

| Action | Description |
|---|---|
| 0 | No-op |
| 1–5 | Schedule task from queue slot 1–5 |
| 6 | Scale up (add CPU core) |
| 7 | Scale down (remove CPU core) |

### Reward Function

| Signal | Value |
|---|---|
| Task completed | +20 × priority |
| Task expired (deadline missed) | −8 |
| Queue length penalty (per step) | −0.1 × queue length |
| Idle CPU penalty (per step) | −0.02 × idle cores |
| Scale up cost | −0.5 |
| Scale down cost | −0.2 |

---

## Project Structure

```
adaptive_scaling_rl/
├── agents/
│   └── ddqn.py              # DDQN agent (online + target network, replay buffer)
├── env/
│   └── task_env.py          # Phase-aware task scheduling environment
├── baseline/
│   ├── fifo.py              # FIFO scheduling baseline
│   └── threshold_scaling.py # Kubernetes HPA-equivalent reactive baseline
├── experiments/
│   ├── run_ddqn.py          # Training entry point (supports --seed argument)
│   ├── run_baseline.py      # Baseline evaluation runner
│   └── plot_rewards.py      # Training curve visualisation (mean ± std)
├── results/                 # Saved reward arrays (per seed)
├── docs/                    # Design documents and FoU research reports
├── requirements.txt
└── Makefile                 # One-command training, baseline, and plotting
```

---

## Quick Start

### Install

```bash
git clone https://github.com/your-org/adaptive_scaling_rl.git
cd adaptive_scaling_rl
pip install -r requirements.txt
```

### Train (single seed)

```bash
python experiments/run_ddqn.py
# or with a specific seed:
python experiments/run_ddqn.py 3
```

### Train all 10 seeds

```bash
make train_all
```

### Evaluate baselines

```bash
make baseline
```

### Plot results

```bash
make plot_all
```

---

## Experimental Results

Training curve — mean ± 1 std across 10 independent seeds (5,000 episodes each):

- Convergence: ~episode 1,000 (DDQN crosses FIFO baseline)
- Final performance: ~4,400 mean reward
- Standard deviation across seeds: narrow throughout — highly reproducible
- No dip curve: resolved by aligning target network update frequency with burst-quiet cycle length

---

## Baseline Comparison

### Why Kubernetes HPA-equivalent performs worse than random

The threshold-based reactive policy scales CPU up when queue length exceeds a fixed threshold. In a burst-quiet patterned environment this causes consistent over-provisioning during quiet phases (idle CPU penalty accumulates) and under-provisioning at burst onset (deadline misses spike). The net effect is a mean reward of −502 — significantly below the random policy at 1,443. This is the core empirical finding that motivates ProScale: **reactive scaling is actively harmful in structured workloads**, and proactive pattern-aware learning is required.

---

## Research Background

This project is developed as part of a Norwegian FoU (Forskning og Utvikling) R&D programme investigating the application of reinforcement learning to proactive cloud resource management. For full methodology, experimental protocol, and technical documentation see `docs/`.

**Research questions:**
- RQ1: Can a DDQN agent learn a scheduling policy that consistently outperforms reactive heuristic baselines?
- RQ2: How do hyperparameters (epsilon decay, target update frequency) affect convergence stability in phase-switching environments?
- RQ3: Is the learned policy stable and reproducible across random seeds?

---

## Roadmap

- [x] Phase-aware simulation environment
- [x] DDQN agent with stable convergence (10-seed validated)
- [x] Baseline comparison (random, FIFO, HPA-equivalent)
- [ ] Business KPI metrics module (SLA violation rate, utilisation %)
- [ ] Prometheus / CloudWatch metrics adapter (real workload integration)
- [ ] FastAPI serving endpoint (`/predict`, `/status`)
- [ ] PPO algorithm comparison
- [ ] Docker deployment
- [ ] Online learning with continuous policy update

---

## License

This project is proprietary research software developed under the ProScale R&D programme.
© RANovaX. All rights reserved.
