# Donkey Kong AI

PPO-based AI agent that learns to play Donkey Kong (NES) using stable-baselines3 and gym-retro.

---

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU recommended (RTX 4070 or similar) — AMD GPUs are not supported by PyTorch CUDA

---

## Setup

### 1. Install PyTorch with CUDA (NVIDIA GPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `True  NVIDIA GeForce RTX 4070`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. ROM setup

You need a Donkey Kong NES ROM. Place it in the project root with one of these names:
- `Donkey Kong (World) (Rev 1).nes`
- `Donkey Kong.nes`

Then run:
```bash
python setup_rom.py
```

This copies the ROM into the `retro_data/` integration folder and verifies the environment loads.

---

## Training

```bash
# Fresh training run (1M steps — increase for better results)
python main.py --algo ppo --timesteps 5000000

# Continue from a saved checkpoint
python main.py --algo ppo --timesteps 5000000 --load-model saved_models/ppo_final

# Run without ghost viewer (faster, good for overnight)
python main.py --algo ppo --timesteps 5000000 --no-ghost
```

Checkpoints are saved every 50k steps to `saved_models/`. The ghost viewer window shows:
- Coloured dots = each agent's current position
- **C** key = toggle height chart (shows how high each agent climbs per episode)

---

## Evaluation

Watch the trained model play:
```bash
python main.py --algo ppo --eval-only --load-model saved_models/ppo_final --render --num-envs 1
```

Each episode prints the reward and highest platform reached.

---

## Project structure

```
environment/
  donkey_kong_env.py   # Core env — reward, frame stack, action mapping
  gym_wrapper.py       # Gymnasium wrapper for stable-baselines3
  ghost_viewer.py      # Live pygame visualiser
training/
  train_ppo.py         # PPO training loop
evaluation/
  metrics.py           # Episode tracking
  plot_results.py      # Reward/height plots
tools/
  debug_detection.py   # Interactive Mario/barrel detection debugger
retro_data/            # Game integration files (ROM not included)
```

---

## Notes

- The ROM file is not included in this repo (copyright). You must provide your own.
- Saved models are not committed to git. Share `.zip` files manually if needed.
- Training on CPU is slow but works. A GPU cuts training time by 10–20×.
