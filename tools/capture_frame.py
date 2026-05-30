"""
Captures one frame from the game, saves it as frame.png, then prints the HSV
value at any pixel you click on. Use this to find Mario's exact skin colour.

Run:  python tools/capture_frame.py
Then click on Mario's face, then a barrel, then press Q to quit.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import retro

INTEGRATION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'retro_data'))
retro.data.Integrations.add_custom_path(INTEGRATION_PATH)

env = retro.make(
    game='DonkeyKong-Nes',
    state='1Player.GameA',
    inttype=retro.data.Integrations.CUSTOM_ONLY,
)

obs, _ = env.reset()

# Advance a few seconds so gameplay has started
BTN_NONE = [0]*9
for _ in range(120):
    result = env.step(BTN_NONE)
    obs = result[0]

env.close()

# obs is RGB; save as PNG (cv2 needs BGR)
frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
cv2.imwrite('frame.png', frame_bgr)
print(f'Saved frame.png  ({obs.shape[1]}×{obs.shape[0]} px)')
print('Click pixels in the window — HSV values will print here. Press Q to quit.\n')

frame_hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)

# Display with pygame (cv2 GUI not available in headless install)
import pygame
pygame.init()

H, W = obs.shape[:2]
SCALE = 3
screen = pygame.display.set_mode((W * SCALE, H * SCALE))
pygame.display.set_caption('Click pixels — Q to quit')
font = pygame.font.SysFont(None, 20)

# Convert RGB numpy → pygame surface
surf = pygame.surfarray.make_surface(obs.transpose(1, 0, 2))
scaled = pygame.transform.scale(surf, (W * SCALE, H * SCALE))

running = True
while running:
    screen.blit(scaled, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            px, py = mx // SCALE, my // SCALE
            px = min(px, W - 1)
            py = min(py, H - 1)
            r, g, b = obs[py, px]
            h, s, v = frame_hsv[py, px]
            print(f'  pixel ({px:3d},{py:3d})  RGB=({r:3d},{g:3d},{b:3d})  HSV=({h:3d},{s:3d},{v:3d})')

    pygame.display.flip()

pygame.quit()
