"""
Run this script once before training to import the ROM into gym-retro
and verify the environment loads correctly.

Usage:
    python setup_rom.py
"""

import hashlib
import os
import shutil
import sys


ROM_FILENAME = 'Donkey Kong (World) (Rev 1).nes'
GAME_NAME    = 'DonkeyKong-Nes'


def sha1(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def find_retro_data_path() -> str:
    import retro
    return os.path.join(os.path.dirname(retro.__file__), 'data', 'stable')


def import_rom():
    rom_path = os.path.join(os.path.dirname(__file__), ROM_FILENAME)
    if not os.path.exists(rom_path):
        print(f'ERROR: ROM not found at:\n  {rom_path}')
        print(f'Place "{ROM_FILENAME}" in the project directory.')
        sys.exit(1)

    integration_dir = os.path.join(
        os.path.dirname(__file__), 'retro_data', GAME_NAME
    )
    os.makedirs(integration_dir, exist_ok=True)

    # Copy ROM into integration dir so stable-retro can find it by SHA
    rom_dest = os.path.join(integration_dir, 'rom.nes')
    if not os.path.exists(rom_dest):
        shutil.copy2(rom_path, rom_dest)
        print(f'Copied ROM → {rom_dest}')
    else:
        print(f'ROM already in integration dir.')

    # Write sha hash file
    sha_path = os.path.join(integration_dir, 'rom.sha')
    digest = sha1(rom_dest)
    with open(sha_path, 'w') as f:
        f.write(digest + '\n')
    print(f'Written rom.sha ({digest[:10]}…) → {sha_path}')


def verify_env():
    import retro

    integration_dir = os.path.join(os.path.dirname(__file__), 'retro_data')
    retro.data.Integrations.add_custom_path(integration_dir)

    # Prefer custom integration; fall back to default if rom was imported
    for inttype in [retro.data.Integrations.CUSTOM_ONLY,
                    retro.data.Integrations.DEFAULT]:
        try:
            env = retro.make(game=GAME_NAME, inttype=inttype)
            result = env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            env.close()
            print(f'\nEnvironment loaded successfully!')
            print(f'  Integration : {inttype}')
            print(f'  Screen shape: {obs.shape}')
            print(f'  Action space: {env.action_space}')
            return
        except Exception as e:
            continue

    # Last resort: list available games to help debug
    print('\nFailed to load environment. Available games containing "Donkey":')
    for g in retro.data.list_games():
        if 'Donkey' in g or 'donkey' in g:
            print(f'  {g}')
    print('\nUpdate GAME_NAME in setup_rom.py to match one of the above.')


if __name__ == '__main__':
    import_rom()
    verify_env()
