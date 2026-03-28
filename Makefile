PYTHON = python3
PIP = pip3

.PHONY: install test fifo threshold train plot clean

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) test_env.py

fifo:
	$(PYTHON) -m baseline.fifo

threshold:
	$(PYTHON) -m baseline.threshold_scaling

train:
	$(PYTHON) -m experiments.run_dqn

plot:
	$(PYTHON) experiments/plot_rewards.py

clean:
	rm -f results/*.npy results/*.png
