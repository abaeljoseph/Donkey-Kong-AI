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


ROM_CONFIGS = [
    {
        'candidates': ['Donkey Kong Original Edition.nes'],
        'game_name':  'DonkeyKongOriginalEdition-Nes',
    },
    {
        'candidates': ['Donkey Kong (World) (Rev 1).nes', 'Donkey Kong.nes'],
        'game_name':  'DonkeyKong-Nes',
    },
]


def sha1(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def find_retro_data_path() -> str:
    import retro
    return os.path.join(os.path.dirname(retro.__file__), 'data', 'stable')


def import_rom():
    project_dir = os.path.dirname(__file__)
    found_any = False

    for config in ROM_CONFIGS:
        game_name = config['game_name']
        rom_path = None
        for candidate in config['candidates']:
            path = os.path.join(project_dir, candidate)
            if os.path.exists(path):
                rom_path = path
                break
        if rom_path is None:
            continue

        found_any = True
        print(f'Found ROM: {os.path.basename(rom_path)} → {game_name}')

        integration_dir = os.path.join(os.path.dirname(__file__), 'retro_data', game_name)
        os.makedirs(integration_dir, exist_ok=True)

        rom_dest = os.path.join(integration_dir, 'rom.nes')
        if not os.path.exists(rom_dest):
            shutil.copy2(rom_path, rom_dest)
            print(f'  Copied ROM → {rom_dest}')
        else:
            print(f'  ROM already in integration dir.')

        sha_path = os.path.join(integration_dir, 'rom.sha')
        digest = sha1(rom_dest)
        with open(sha_path, 'w') as f:
            f.write(digest + '\n')
        print(f'  Written rom.sha ({digest[:10]}…)')

    if not found_any:
        print('ERROR: No ROM found. Place one of these files in the project directory:')
        for config in ROM_CONFIGS:
            for c in config['candidates']:
                print(f'  {c}')
        sys.exit(1)


def verify_env():
    import retro

    integration_dir = os.path.join(os.path.dirname(__file__), 'retro_data')
    retro.data.Integrations.add_custom_path(integration_dir)

    verified = []
    for config in ROM_CONFIGS:
        game_name = config['game_name']
        for inttype in [retro.data.Integrations.CUSTOM_ONLY,
                        retro.data.Integrations.DEFAULT]:
            try:
                env = retro.make(game=game_name, inttype=inttype)
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result
                env.close()
                verified.append((game_name, obs.shape))
                break
            except Exception:
                continue

    if verified:
        print('\nEnvironments loaded successfully:')
        for name, shape in verified:
            print(f'  {name}  screen shape: {shape}')
    else:
        print('\nFailed to load any environment. Available games containing "Donkey":')
        for g in retro.data.list_games():
            if 'Donkey' in g or 'donkey' in g:
                print(f'  {g}')


if __name__ == '__main__':
    import_rom()
    verify_env()
