# Donkey Kong AI

A reinforcement learning agent that learns to play Donkey Kong (NES) by itself. It uses PPO (a type of AI training algorithm) and a convolutional neural network to watch the game screen and figure out what buttons to press.

---

## STOP — Read this first

**You cannot run this on regular Windows.** You must use WSL2 (a Linux environment inside Windows). If you skip this and try to run it normally, it will not work.

---

## Step 1 — Install WSL2

1. Open **PowerShell as Administrator** (right-click the Start menu → Windows PowerShell (Admin))
2. Run:
   ```powershell
   wsl --install
   ```
3. Restart your PC when it asks
4. After restart, **Ubuntu** will open and ask you to create a Linux username and password — do that
5. All future commands should be typed in the **Ubuntu** terminal, not PowerShell

---

## Step 2 — Install Python inside WSL

In your Ubuntu terminal:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv -y
```

Check it worked:
```bash
python3 --version
```
Should show Python 3.10 or higher.

---

## Step 3 — Install CUDA (NVIDIA GPU only)

If you have an NVIDIA GPU (e.g. RTX 4070), install PyTorch with GPU support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Check your GPU is detected:
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print something like `True  NVIDIA GeForce RTX 4070`.

If you have no NVIDIA GPU (e.g. AMD or no GPU), skip this step — it will run on CPU instead (slower but works).

---

## Step 4 — Clone the repo inside WSL

```bash
cd ~
git clone <your-repo-url>
cd "Donkey Kong AI"
```

> Important: clone inside the Linux filesystem (`~/`), NOT inside `/mnt/c/...`. Working from the Windows drive is much slower.

---

## Step 5 — Install all Python dependencies

```bash
pip install -r requirements.txt
```

This installs everything the project needs including:
- `stable-baselines3` — the PPO training library
- `gymnasium` — the RL environment framework
- `stable-retro` — NES emulator wrapper
- `opencv-python` — frame processing
- `pygame` — the ghost viewer window
- `numpy`, `torch`, `tensorboard`

---

## Step 6 — Get the ROM file

You need a Donkey Kong NES ROM file. The ROM is not included in this repo (copyright).

1. Get a file named either:
   - `Donkey Kong (World) (Rev 1).nes`
   - `Donkey Kong.nes`

2. Copy it from Windows into WSL:
   ```bash
   cp /mnt/c/Users/YourName/Downloads/"Donkey Kong (World) (Rev 1).nes" ~/
   ```
   Replace `YourName` with your actual Windows username.

3. Move it into the project folder:
   ```bash
   mv "Donkey Kong (World) (Rev 1).nes" ~/"Donkey Kong AI"/
   ```

---

## Step 7 — Set up the ROM

```bash
python setup_rom.py
```

This registers the ROM with the emulator and checks everything loads correctly. You should see `Environment loaded successfully!` at the end.

---

## Step 8 — Train the AI

```bash
python main.py --algo ppo --timesteps 5000000
```

This starts training across 8 parallel game environments. A ghost viewer window will open showing all agents playing in real time.

To continue training from a saved checkpoint:
```bash
python main.py --algo ppo --timesteps 5000000 --load-model saved_models/ppo_final
```

To train without the ghost viewer (faster, good for overnight):
```bash
python main.py --algo ppo --timesteps 5000000 --no-ghost
```

---

## Step 9 — Watch the trained model play

```bash
python main.py --algo ppo --eval-only --load-model saved_models/ppo_final --render --num-envs 1
```

This opens the game window and shows the AI playing. Each episode prints the reward and highest platform Mario reached.

---

## Ghost Viewer — what you're looking at

The ghost viewer window opens automatically during training.

| What you see | What it means |
|---|---|
| Coloured dots (E0–E7) | Each of the 8 agents and where they are on screen |
| Star ring around a dot | The agent with the highest reward this episode |
| Orange dot labelled B | A barrel detected rolling toward Mario |
| Green horizontal line | The win line — Mario needs to reach above this |
| Red overlay top strip | HUD zone — score display, ignored by the AI |
| Red box top-left | DK zone — Donkey Kong is excluded so he's not confused with Mario |
| Chart (bottom-right) | Height chart — shows how high each agent is climbing |

**Press C** during training to toggle the height chart on and off.

The height chart has two panels:
- **Left** — history graph showing the best height reached per episode over time (you want this trending upward)
- **Right** — live coloured bars showing each agent's current height in the current episode in NES pixels

---

## Debug tool — tune detection

If Mario or barrels are being detected incorrectly, run this interactive tool:
```bash
python tools/debug_detection.py
```

| Key | What it does |
|---|---|
| H then drag | Draw the HUD exclusion zone |
| D then drag | Draw the DK exclusion zone |
| Z | Toggle overlays on/off to see raw colours |
| P | Print zone values to paste into code |
| F | Advance 60 more game frames |
| R | Reset back to the start |
| Click | Print the RGB and HSV colour of that pixel |

---

## How the AI works

The agent watches the screen and decides what button to press every 8 game frames.

**What it sees:** 4 stacked greyscale frames (84×84 pixels) + a 5th channel showing where the ladders are (highlighted in teal). This gives the CNN both motion context and explicit ladder awareness.

**What it can do:** 8 actions — do nothing, left, right, up (climb ladder), down, jump, jump+left, jump+right.

**How it learns:** PPO collects experience from 8 parallel games, then stops to update the network every 256 steps per environment. The reward function guides it:

| Situation | Reward |
|---|---|
| Climbing upward | +3 to +10 per NES pixel (scales higher near the top) |
| Climbing a ladder (pressing Up near teal) | +1.0 |
| Near a ladder | +0.1 |
| Barrel within 15 pixels | up to -0.5 |
| Each step taken | -0.3 (discourages idling) |
| Death | -3.0 |
| Game over | -5.0 |
| Winning (reaching Pauline) | +500 |

---

## Project structure

```
environment/
  donkey_kong_env.py   # Core environment — reward, actions, frame processing
  gym_wrapper.py       # Wraps it for stable-baselines3
  ghost_viewer.py      # Live visualiser with height chart
training/
  train_ppo.py         # Training and evaluation loop
evaluation/
  metrics.py           # Tracks reward, steps, deaths per episode
  plot_results.py      # Generates reward/height graphs after training
tools/
  debug_detection.py   # Interactive colour detection debugger
retro_data/            # NES game integration files (ROM not included)
saved_models/          # Where checkpoints are saved during training
```

---

## Team

| Person | Role |
|---|---|
| Person 1 | Environment wrapper, reward shaping, colour detection, ghost viewer |
| Person 2 | CNN feature extractor |
| Person 3 | DQN agent and replay buffer |
| Person 4 | Evaluation metrics and result plots |

---

## Notes

- The ROM is not in this repo — you must provide your own copy
- Saved models are not committed to git — share `.zip` files manually
- At least 5 million timesteps recommended before the agent starts climbing consistently
- GPU training is 10–20x faster than CPU
