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
- `pygame` — the ghost viewer and debug tools
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

**Train headless — fastest, recommended for overnight runs:**
```bash
python main.py --algo ppo --timesteps 10000000 --num-envs 8
```

**Train and watch — opens game window and ghost viewer:**
```bash
python main.py --algo ppo --timesteps 10000000 --num-envs 8 --render
```

**With platform checkpoints (curriculum learning — Mario drills each platform):**
```bash
python main.py --algo ppo --timesteps 10000000 --num-envs 8 --checkpoints
```

**Continue training from a saved checkpoint:**
```bash
python main.py --algo ppo --timesteps 10000000 --load-model saved_models/ppo_XXXXXX_steps
```
Replace `ppo_XXXXXX_steps` with the filename of your latest checkpoint (without `.zip`). To find it:
```bash
ls saved_models/ppo_*.zip | sort -V | tail -3
```

If the model path is wrong or the file doesn't exist, the script will tell you and exit — it will not silently start a new model.

**Choosing how many environments (`--num-envs`):**

More environments = more game data collected at once, but each environment needs a CPU core. Running more environments than you have cores slows everything down.

Check how many cores your CPU has:
```bash
nproc
```
Set `--num-envs` to that number. 8 is a safe default for most machines.

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

If the model was trained with a different observation shape (e.g. a different number of channels), the script will detect the mismatch and tell you rather than crashing or running incorrectly.

---

## Command Reference

### Common flags

| Flag | What it does |
|---|---|
| `--algo ppo` | Use the PPO algorithm (required) |
| `--timesteps N` | Total number of game steps to train for (e.g. `10000000`) |
| `--num-envs N` | Number of parallel games running at once — set to your CPU core count |
| `--render` | Open the game window and ghost viewer (for both training and eval) |
| `--load-model PATH` | Load a saved model — exits with an error if the path is wrong |
| `--eval-only` | Run the model without training — just watch it play |
| `--ppo-save-freq N` | Save a checkpoint every N training steps (default: 50000) |

### Spawn / checkpoint flags

| Flag | What it does |
|---|---|
| *(nothing)* | Mario always spawns at the ground floor |
| `--checkpoints` | Enable platform checkpoints — Mario spawns randomly across all saved platforms, weighted toward higher ones |
| `--force-highest` | Always spawn at the highest saved platform — enables checkpoints automatically |

### Example commands

| Goal | Command |
|---|---|
| Train from scratch (headless) | `python main.py --algo ppo --timesteps 10000000 --num-envs 8` |
| Train and watch | `python main.py --algo ppo --timesteps 10000000 --num-envs 8 --render` |
| Train with curriculum checkpoints | `python main.py --algo ppo --timesteps 10000000 --num-envs 8 --checkpoints` |
| Continue training a saved model | `python main.py --algo ppo --timesteps 10000000 --load-model saved_models/ppo_XXXXXX_steps` |
| Watch the final model play | `python main.py --algo ppo --eval-only --load-model saved_models/ppo_final --render --num-envs 1` |
| Watch from the highest checkpoint | `python main.py --algo ppo --eval-only --load-model saved_models/ppo_final --render --num-envs 1 --force-highest` |

---

## Ghost Viewer — what you're looking at

The ghost viewer opens when you pass `--render` during training. It does not open during headless training.

| What you see | What it means |
|---|---|
| Coloured dots (E0–E7) | Each agent and where they are on screen |
| Star ring around a dot | The agent with the highest reward this episode |
| Orange dot labelled B | A barrel tracked by OAM sprite data |
| Fire dot labelled F | A fireball tracked by OAM sprite data |
| Green horizontal line | The win line — Mario needs to reach above this |
| Cyan box (L0, L1…) | Intact ladder segment |
| Red box (BROKEN) | Broken ladder segment — AI is penalised for climbing these |
| `Lv1 Barrels` in the HUD | Current level and stage read from NES RAM |
| Chart (bottom-right) | Height chart — how high each agent is climbing |
| Chart (bottom-left) | Reward history — step reward over time per agent |

### Ghost Viewer key controls

| Key | What it does |
|---|---|
| **C** | Toggle height chart on/off |
| **R** | Toggle reward history chart on/off |
| **L** | Toggle ladder labels and broken zone rectangles on/off |
| **Z** | Toggle HUD and DK exclusion zone overlays on/off |
| **D** | Toggle barrel and fire danger markers on/off |

---

## Debug tool — interactive RAM inspector

Used to play the game manually and inspect what the AI can see. Useful for verifying that sprite positions, broken ladder zones, and RAM values are being read correctly.

```bash
python tools/test_broken_ladder.py
```

The window has two sections: the game on the left and a live data panel on the right.

**Game area (left):**
- Pink crosshair + green dot — Mario's exact position from OAM sprite data
- Red crosshairs — active barrels (only visible when DK has thrown them)
- Orange crosshairs — active fireballs
- Yellow crosshairs — hammers
- Red rectangles — broken ladder zones

**Data panel (right):**
- Game state: score, lives, stage, level, bonus timer
- Mario X/Y coordinates
- Broken zone status (turns red when Mario is inside one)
- Barrel count + exact OAM coordinates of each active barrel
- Fire count + exact OAM coordinates of each active fire
- Step count, per-step reward, cumulative reward
- All detected broken zone rectangles

**Controls:**

| Key | What it does |
|---|---|
| **Arrow keys** | Move / climb ladders |
| **Space** | Jump |
| **Q / Escape** | Quit (prints bonus timer analysis on exit) |

When you quit, the tool analyses the full RAM recording and reports which address is the bonus/time-remaining counter — useful for identifying new RAM locations.

---

## How the AI works

The agent watches the screen and decides what button to press every 8 game frames.

**What it sees — 7 channels (84×84 pixels each):**

| Channel | Content | Source |
|---|---|---|
| 0–3 | Grayscale frames × 4 | Last 4 rendered NES frames, stacked for motion context |
| 4 | Ladder positions | HSV colour detection — computed once per episode (ladders are static) |
| 5 | Barrel positions | OAM sprite RAM — exact positions painted as blobs, updated every step |
| 6 | Fire positions | OAM sprite RAM — exact positions painted as blobs, updated every step |

The 4 stacked frames let the CNN infer movement — a barrel appearing in different positions across the 4 frames tells it the barrel is rolling and which direction it is heading.

**What it can do:** 8 actions — do nothing, left, right, up (climb ladder), down, jump, jump+left, jump+right.

**How it learns:** PPO collects experience from parallel games and updates the network every 512 steps per environment. Rewards are normalised for stable training.

**Reward function:**

| Situation | Reward |
|---|---|
| Climbing to a new height record this life | +3 to +10 per NES pixel (scales higher near the top — resets on death so every life gets the same reward signal) |
| Pressing UP while moving up with a ladder visible | +0.5 (active climbing bonus — small so farming gives only ~+0.10/step over idling, dominated by height reward) |
| Teal ladder visible anywhere on screen | +0.3 (passive — keeps agent near ladders) |
| Each step taken | −0.05 (discourages idling) |
| Pressing UP inside a broken ladder zone | −2.0 (active bonus also suppressed to +0.3) |
| Barrel within 15 NES pixels | up to −0.2 |
| Fire within 20 NES pixels | up to −0.2 |
| Death | −3.0 |
| Game over | −5.0 |
| Winning (reaching Pauline) | +500 + remaining steps × 0.5 |

**How sprite detection works:**

Mario's position, barrel proximity, and fire proximity are all read directly from NES OAM (Object Attribute Memory) shadow at RAM address 512. Each sprite is 4 bytes: Y, tile, attributes, X. The confirmed slot layout for Donkey Kong NES is:

| OAM slots | Object |
|---|---|
| 0–3 | Mario |
| 4–7 | Fire 1 |
| 8–11 | Fire 2 |
| 12–51 | Barrels (4 sprites per barrel, up to 10 barrels) |
| 52–55 | Hammers |

This replaces all HSV colour scanning for hazard detection — OAM reads are instant, pixel-perfect, and never produce false positives from background tiles or overlapping sprites.

**Ladder detection** still uses HSV colour detection (teal pixels), but only once per episode at reset time. Ladders are static within a stage so there is no need to re-scan every step.

**Platform checkpoints (opt-in):** The first time Mario safely reaches each platform, the emulator state is saved. Future episodes sample across all saved platforms — weighted toward higher ones — so the agent gets concentrated practice at every transition rather than always climbing from scratch.

**Stage and level detection:** The NES RAM is read every step. Address 83 holds the current stage (1 = Barrels, 3 = Elevator, 4 = Rivets) and address 84 holds the level counter. When the stage register changes from its starting value, the episode ends as a win.

**Broken ladder detection:** At episode start, connected-component analysis finds all broken ladder pairs (two short teal stubs at the same X range separated by a small vertical gap). These coordinates are cached for the episode. Pressing UP inside a broken zone costs −2.0 reward and suppresses the active climbing bonus (+0.5 → +0.3). Height reward still fires normally so the agent can still climb through if needed, but intact ladders are always more rewarding (no penalty, full +0.5 active bonus).

**Model compatibility:** Saved models are tied to the observation shape they were trained with. If you change `OBS_CHANNELS` in `environment/donkey_kong_env.py`, existing saved models will be rejected with a clear error — the script will not silently start fresh or run a model with the wrong input shape.

---

## Robot Arms (Simulated)

Two KUKA IIWA 7-DOF robot arms mirror what the AI is doing in real time:
- **Left arm (blue)** — grips a joystick and tilts it left, right, forward or back to match the AI's directional input
- **Right arm (orange)** — presses a button down whenever the AI jumps. The button turns red while held and yellow when released

### Install the robot arm dependency

```bash
pip install pybullet
```

If `pybullet` is already in `requirements.txt` it will have been installed in Step 5. You can confirm with:

```bash
python -c "import pybullet; print('pybullet OK')"
```

### Run with robot arms

Watch the AI play with both arms moving in a separate 3D window:

```bash
python main.py --algo ppo --eval-only --load-model previous_archive/ppo59a_20m --num-envs 1 --arm-gui --render
```

NO GPU:
```bash 
python main.py --algo ppo --eval-only --load-model previous_archive/ppo59a_20m --num-envs 1 --arm-gui --arm-renderer software --render
```

Watch arms only (no game window):

```bash
python main.py --algo ppo --eval-only --load-model previous_archive/ppo59a_20m --num-envs 1 --arm-gui
```

Train with arms running in the background (headless, no GUI):

```bash
python main.py --algo ppo --timesteps 10000000 --num-envs 8 --arm
```

### Robot arm flags

| Flag | What it does |
|---|---|
| `--arm-gui` | Open the 3D PyBullet window showing both KUKA arms moving |
| `--arm` | Run the arms headless (no window) — useful during training |

### How the arms work

Each AI action maps to a physical movement:

| Action | Joystick arm | Button arm |
|---|---|---|
| NOOP | Joystick centred | Arm raised |
| LEFT | Joystick tilts left | Arm raised |
| RIGHT | Joystick tilts right | Arm raised |
| UP | Joystick tilts forward | Arm raised |
| DOWN | Joystick tilts back | Arm raised |
| JUMP | Joystick centred | Presses button down |
| JUMP + LEFT | Joystick tilts left | Presses button down |
| JUMP + RIGHT | Joystick tilts right | Presses button down |

---

## Project structure

```
environment/
  donkey_kong_env.py   # Core environment — reward, actions, OAM detection, frame processing
  gym_wrapper.py       # Wraps it for stable-baselines3
  ghost_viewer.py      # Live visualiser with height and reward charts
  pybullet_arm.py      # Dual KUKA IIWA arm simulation driven by AI actions
training/
  train_ppo.py         # Training and evaluation loop with model validation
evaluation/
  metrics.py           # Tracks reward, steps, deaths per episode
  plot_results.py      # Generates reward/height graphs after training
tools/
  test_broken_ladder.py  # Interactive debug tool — play manually, inspect OAM data live
  debug_detection.py     # Colour detection inspector (for ladder HSV tuning)
retro_data/            # NES game integration files (ROM not included)
saved_models/          # Where checkpoints are saved during training
```

---

## Team

| Person | Role |
|---|---|
| Person 1 | Environment wrapper, reward shaping, OAM sprite detection, ghost viewer |
| Person 2 | CNN feature extractor |
| Person 3 | DQN agent and replay buffer |
| Person 4 | Evaluation metrics and result plots |

---

## Notes

- The ROM is not in this repo — you must provide your own copy named `Donkey Kong.nes`
- Saved models are not committed to git — share `.zip` files manually
- At least 10 million timesteps recommended before the agent starts climbing consistently
- GPU training (NVIDIA only) is 10–20× faster than CPU — change `device='cpu'` to `device='cuda'` in `training/train_ppo.py`
- Saved models trained before OAM-based detection was added (pre-7-channel) are incompatible — the script will detect this and tell you rather than crashing
- To expand to 8 observation channels (adding a hammer channel): set `OBS_CHANNELS = 8` in `donkey_kong_env.py` and uncomment the hammer line in `_get_state()` — requires retraining from scratch
- The level counter (NES RAM address 84) is 0-indexed internally — add 1 before displaying so L1 matches what the game shows
- The stage register (NES RAM address 83) cycles 1 → 3 → 4 → 1, not 1 → 2 → 3 — slot 2 is reserved and skipped by the ROM
- Each time you continue training from a checkpoint, TensorBoard creates a new run line — the previous run's graph is preserved separately
- If `height/mean` in TensorBoard is flat after 300k steps, check `environment/donkey_kong_env.py` — the reward signal may need tuning


## Running the enviorment with Robot arms

- Install:
pip install pybullet

Run (arms + game window together):
python main.py --algo ppo --eval-only --load-model previous_archive/ppo59a_20m --num-envs 1 --arm-gui --render

