"""
Entry point.

Train headless (default):
    python main.py --algo ppo --timesteps 10000000 --num-envs 8

Train and watch (ghost viewer + game window):
    python main.py --algo ppo --timesteps 10000000 --num-envs 8 --render

Continue training a checkpoint:
    python main.py --algo ppo --timesteps 10000000 --num-envs 8 --load-model saved_models/ppo_5000000_steps

Watch a saved model:
    python main.py --algo ppo --eval-only --load-model saved_models/ppo_final --render --num-envs 1

PyBullet simulated arms:
    python main.py --algo ppo --arm-gui

Real UR3 arms (calibrate first with tools/calibrate_arms.py):
    python main.py --algo ppo --eval-only --load-model <ckpt> --real-arms --num-envs 1
"""

import argparse
import os

from evaluation.plot_results import plot_all_metrics


def parse_args():
    p = argparse.ArgumentParser(description='Donkey Kong AI Agent')
    p.add_argument('--algo',        type=str,   default='dqn', choices=['dqn', 'ppo'],
                   help='Algorithm: dqn (custom) or ppo (stable-baselines3)')
    # DQN args
    p.add_argument('--episodes',    type=int,   default=1000,  help='DQN: number of episodes')
    p.add_argument('--save-freq',   type=int,   default=100,   help='DQN: save every N episodes')
    # PPO args
    p.add_argument('--timesteps',   type=int,   default=1_000_000, help='PPO: total env timesteps')
    p.add_argument('--ppo-save-freq', type=int, default=50_000,    help='PPO: checkpoint every N steps')
    p.add_argument('--checkpoints',    action='store_true', help='Enable checkpoint spawning during training/eval')
    p.add_argument('--force-highest',  action='store_true', help='Always spawn at the highest saved checkpoint (implies --checkpoints)')
    # Shared
    p.add_argument('--num-envs',    type=int,   default=24,  help='Parallel environments')
    p.add_argument('--render',      action='store_true', help='Render game window (env 0 only)')
    p.add_argument('--arm',         action='store_true', help='Enable PyBullet simulated arm')
    p.add_argument('--arm-gui',     action='store_true', help='Show PyBullet arm GUI (implies --arm)')
    p.add_argument('--real-arms',   action='store_true', help='Use real UR3 arms (reads robot_config.json)')
    p.add_argument('--load-model',  type=str,   default=None,
                   help='DQN: path to .pt file; PPO: path without .zip extension')
    p.add_argument('--eval-only',   action='store_true')
    p.add_argument('--save-dir',    type=str,   default='saved_models')
    p.add_argument('--metrics-out', type=str,   default='results/metrics.json')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    if args.algo == 'ppo':
        from training.train_ppo import train_ppo
        metrics = train_ppo(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            render=args.render,
            use_arm=args.arm or args.arm_gui,
            arm_gui=args.arm_gui,
            use_real_arms=args.real_arms,
            load_model=args.load_model,
            eval_only=args.eval_only,
            save_dir=args.save_dir,
            save_freq=args.ppo_save_freq,
            ghost_viewer=args.render,
            use_checkpoints=args.checkpoints or args.force_highest,
            force_highest=args.force_highest,
        )
    else:
        from training.train import train
        metrics = train(
            num_episodes=args.episodes,
            num_envs=args.num_envs,
            render=args.render,
            arm_gui=args.arm_gui,
            load_model=args.load_model,
            eval_only=args.eval_only,
            save_dir=args.save_dir,
            save_freq=args.save_freq,
        )

    metrics.save(args.metrics_out)
    plot_all_metrics(metrics)


if __name__ == '__main__':
    main()
