"""
Calibration helper for both UR3 arms.

Puts each robot into freedrive mode so you can physically jog it to the
correct position, then captures the TCP pose and saves it to robot_config.json.

Run once before using --real-arms:
    python tools/calibrate_arms.py

Requirements:
    pip install ur-rtde
"""

import json
import os
import sys
import time

try:
    import rtde_control
    import rtde_receive
except ImportError:
    sys.exit('ur_rtde not installed.  Run:  pip install ur-rtde')

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'robot_config.json')


def load_config():
    if os.path.exists(CFG_PATH):
        with open(CFG_PATH) as f:
            return json.load(f)
    return {
        'joystick_arm': {'ip': '192.168.1.2', 'home_pose': None},
        'button_arm':   {'ip': '192.168.1.3', 'rest_pose': None},
    }


def save_config(cfg):
    with open(CFG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'\nSaved → {CFG_PATH}')


def get_ip(label: str, current: str) -> str:
    val = input(f'{label} IP [{current}]: ').strip()
    return val if val else current


def capture_pose(label: str, ip: str, instructions: str) -> list:
    """
    Connects to the robot, enables freedrive, waits for the user to jog
    the arm into position, then captures and returns the TCP pose.
    """
    print(f'\n--- {label} ---')
    print(f'Connecting to {ip} …')
    ctrl = rtde_control.RTDEControlInterface(ip)
    recv = rtde_receive.RTDEReceiveInterface(ip)
    print('Connected.')

    print(f'\n{instructions}')
    print('Press Enter to enable freedrive and jog the arm into position …')
    input()

    ctrl.freedriveMode()
    print('Freedrive ON  — move the arm to the target position.')
    print('Press Enter when ready to capture the pose …')
    input()

    pose = recv.getActualTCPPose()
    ctrl.endFreedriveMode()
    print(f'Captured pose: {[round(v, 4) for v in pose]}')

    ctrl.stopScript()
    ctrl.disconnect()
    recv.disconnect()
    return list(pose)


def main():
    print('UR3 Dual-Arm Calibration')
    print('=' * 40)

    cfg = load_config()

    # --- IPs ---
    print('\nStep 1 — Enter robot IP addresses.')
    cfg['joystick_arm']['ip'] = get_ip('Joystick arm', cfg['joystick_arm'].get('ip', '192.168.1.2'))
    cfg['button_arm']['ip']   = get_ip('Button arm',   cfg['button_arm'].get('ip', '192.168.1.3'))

    # --- Joystick arm ---
    print('\nStep 2 — Calibrate the JOYSTICK arm.')
    cfg['joystick_arm']['home_pose'] = capture_pose(
        label='Joystick arm — centre position',
        ip=cfg['joystick_arm']['ip'],
        instructions=(
            'Grip the joystick with the robot\'s end-effector.\n'
            'Move the arm so the joystick is perfectly centred (no deflection).\n'
            'This is the neutral / NOOP position.'
        ),
    )

    # --- Button arm ---
    print('\nStep 3 — Calibrate the BUTTON arm.')
    cfg['button_arm']['rest_pose'] = capture_pose(
        label='Button arm — rest position (above button)',
        ip=cfg['button_arm']['ip'],
        instructions=(
            'Position the finger directly above the jump button,\n'
            'roughly 20 mm above the surface — the arm will press\n'
            'down by 20 mm from this pose when the AI jumps.'
        ),
    )

    save_config(cfg)
    print('\nCalibration complete.  You can now run:')
    print('  python main.py --algo ppo --eval-only --load-model <ckpt> --real-arms --num-envs 1')


if __name__ == '__main__':
    main()
