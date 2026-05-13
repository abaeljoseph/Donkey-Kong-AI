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

1. Get a file named: `Donkey Kong.nes`

2. Copy it from Windows into WSL:
   ```bash
   cp /mnt/c/Users/YourName/Downloads/"Donkey Kong.nes" ~/
   ```
   Replace `YourName` with your actual Windows username.

3. Move it into the project folder:
   ```bash
   mv "Donkey Kong.nes" ~/"Donkey Kong AI"/
   ```

---

## Step 7 — Set up the ROM

```bash
python setup_rom.py
```

This registers the ROM with the emulator and checks everything loads correctly. You should see `Environment loaded successfully!` at the end.

---

## Step 8 — Activate the virtual environment

Every time you open a new WSL terminal, run this before any other command:
```bash
source .venv/bin/activate
```

You should see `(.venv)` appear at the start of your prompt. If you skip this, Python will not find the installed packages.

---

## Step 9 — Train the AI

**Recommended — fast training, no visualiser (best for overnight runs):**
```bash
python main.py --algo ppo --timesteps 10000000 --no-ghost --num-envs 8
```

**With live ghost viewer (slower, but lets you watch agents play):**
```bash
python main.py --algo ppo --timesteps 10000000 --num-envs 8
```

**Continue training from a saved checkpoint:**
```bash
python main.py --algo ppo --timesteps 10000000 --no-ghost --load-model saved_models/ppo_XXXXXX_steps
```
Replace `ppo_XXXXXX_steps` with the filename of your latest checkpoint (without `.zip`). To find it:
```bash
ls saved_models/ppo_*.zip | sort -V | tail -3
```

**Choosing how many environments (`--num-envs`):**

More environments = more game data collected at once, but each environment needs a CPU core. Running more environments than you have cores slows everything down.

Check how many cores your CPU has:
```bash
nproc
```
Set `--num-envs` to that number. 8 is a safe default for most machines.

**NVIDIA GPU users (e.g. RTX 4070):**
Edit `training/train_ppo.py` and change `device='cpu'` to `device='cuda'` for 10–20× faster training. You can also increase `--num-envs` to match your core count (e.g. `--num-envs 12`).

---

## Step 10 — Monitor training progress

Open a second WSL terminal and run:
```bash
tensorboard --logdir saved_models/tb_logs
```
Then open `http://localhost:6006` in your browser.

**What to watch:**
- `height/mean` — how high Mario is climbing on average. You want this trending upward over time.
- **Smoothed** value (shown in TensorBoard) is more reliable than the raw **Value** — the raw value is noisy and jumps around. Always judge progress by the smoothed line.

TensorBoard data is saved to disk continuously during training. You can close and reopen it at any time — all history will still be there.

---

## Step 11 — Watch the trained model play

```bash
python main.py --algo ppo --eval-only --load-model saved_models/ppo_final --render --num-envs 1
```

This opens the game window and shows the AI playing. Each episode prints the reward and highest platform Mario reached.

---

## Ghost Viewer — what you're looking at

The ghost viewer window opens automatically during training (omit `--no-ghost` to enable it).

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

## Debug tool — tune colour detection

If Mario, barrels, or fire are being detected incorrectly (wrong positions, missing detections, false positives), use this interactive tool to inspect and tune the HSV colour ranges:

```bash
python tools/debug_detection.py
```

The tool opens a window showing the game frame with all detections overlaid. Use it to click pixels, draw exclusion zones, and verify that the right objects are being picked up.

### Controls

| Key / Action | What it does |
|---|---|
| **Click** on a pixel | Prints that pixel's RGB and HSV values to the terminal — use this to sample colours for tuning detection ranges |
| **H** then drag | Draw the HUD exclusion zone (top strip with score display — ignored by the AI) |
| **D** then drag | Draw the DK zone (top-left area where Donkey Kong stands — excluded from barrel detection) |
| **Z** | Toggle detection overlays on/off — lets you see the raw game frame without any markings |
| **P** | Print the current zone coordinates to the terminal — copy these into `donkey_kong_env.py` |
| **F** | Advance 60 game frames forward — use to navigate to different game situations |
| **R** | Reset back to the start of the level |

### How to tune a detection range

1. Run the tool and navigate to a frame where the problem occurs (use **F** to advance frames)
2. Click on a pixel that should be detected but isn't (or one that's being falsely detected)
3. Note the HSV values printed in the terminal
4. Compare with the ranges in `environment/donkey_kong_env.py`:
   - Ladder (teal): H 83–95, S 220–255, V 190–255
   - Barrel (orange): H 10–25, S 160–215, V 220–255
   - Fire: H 12–25, S 50–130, V 220–255
   - Mario skin: H 0–12, S 45–110, V 215–255
5. Adjust the range to include the sampled pixel, then restart the tool to verify

### Common issues

| Problem | Likely cause | Fix |
|---|---|---|
| Mario not detected | Skin/hat/overalls HSV range too narrow | Click Mario pixels, widen the range |
| Barrel detected in wrong place | DK zone not covering stacked barrels at top-left | Press D and redraw the DK zone |
| Fire and barrels confused | Fire S value overlapping barrel S range | Fire has low saturation (S 50–130); barrels have high saturation (S 160+) — check your sampled values |
| Mario detected as barrel | HUD area not excluded | Press H and redraw the HUD zone |

---

## How the AI works

The agent watches the screen and decides what button to press every 8 game frames.

**What it sees:** 7 channels total (84×84 pixels each):
- 4 stacked greyscale frames — motion and layout context over the last 4 decisions
- 1 teal/ladder channel — exact positions of all climbable ladders
- 1 barrel channel — exact positions of rolling barrels
- 1 fire channel — exact positions of fireballs

This gives the CNN explicit spatial awareness of every danger and every ladder on screen. The 4 stacked frames also let it infer movement — a barrel appearing in different positions across the 4 frames tells the CNN it is rolling and which direction it is heading.

**What it can do:** 8 actions — do nothing, left, right, up (climb ladder), down, jump, jump+left, jump+right.

**How it learns:** PPO collects experience from 8 parallel games, then updates the network every 512 steps per environment. Rewards are normalised for stable training. The reward function guides it:

| Situation | Reward |
|---|---|
| Climbing upward | +3 to +10 per NES pixel (scales higher near the top) |
| Actively climbing a ladder (Up + moving up + near teal) | +3.0 |
| Near a ladder | +0.3 |
| Jumping when a barrel or fire is nearby | normal height reward |
| Jumping with no danger nearby | 0 height reward, −0.15 penalty |
| Barrel within 15 pixels | up to −0.2 |
| Fire within 20 pixels | up to −0.2 |
| Each step taken | −0.1 (discourages idling and hesitation) |
| Death | −3.0 |
| Game over | −5.0 |
| Winning (reaching Pauline) | +500 + remaining steps × 0.5 (bonus for finishing fast) |

**Platform checkpoints:** The first time Mario safely reaches each platform, the emulator state is saved. Future episodes are distributed across all saved platforms — weighted toward higher ones — so the agent gets concentrated practice at every transition rather than always climbing from scratch. This also means it drills barrel patterns at each height repeatedly until they become reliable.

**Reward hacking prevention:** The ladder climbing bonus only fires when Mario is actually moving upward — pressing UP while standing still at the base of a ladder gives no bonus.

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

- The ROM is not in this repo — you must provide your own copy named `Donkey Kong.nes`
- Saved models are not committed to git — share `.zip` files manually
- At least 10 million timesteps recommended before the agent starts climbing consistently
- GPU training (NVIDIA only) is 10–20× faster than CPU — change `device='cpu'` to `device='cuda'` in `training/train_ppo.py`
- Old saved models (trained before 7-channel observations were added) are incompatible — delete them and retrain from scratch
- Each time you continue training from a checkpoint, TensorBoard creates a new run line — the previous run's graph is preserved separately
- If `height/mean` in TensorBoard is flat after 300k steps, check `environment/donkey_kong_env.py` — the reward signal may need tuning
