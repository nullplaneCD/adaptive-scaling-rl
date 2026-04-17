PYTHON = python3
PIP = pip3

.PHONY: install test fifo threshold baseline train plot plot_all clean train_all

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) test_env.py

fifo:
	$(PYTHON) -m baseline.fifo

threshold:
	$(PYTHON) -m baseline.threshold_scaling

baseline:
	$(PYTHON) -m experiments.run_baseline
	$(PYTHON) -m baseline.threshold_scaling
	$(PYTHON) -m baseline.fifo

SEED ?= 0

train:
	$(PYTHON) -m experiments.run_ddqn $(SEED)
	$(PYTHON) experiments/plot_rewards.py

plot:
	$(PYTHON) experiments/plot_rewards.py single

plot_all:
	$(PYTHON) experiments/plot_rewards.py multi

train_all:
	for seed in 0 1 2 3 4 5 6 7 8 9; do \
		$(PYTHON) -m experiments.run_ddqn $$seed; \
	done
	$(PYTHON) experiments/plot_rewards.py multi

clean:
	rm -f results/*.npy results/*.png
