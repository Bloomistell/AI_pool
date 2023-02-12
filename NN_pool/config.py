import math

import numpy as np
import pygame

def set_max_resolution():
    infoObject = pygame.display.Info()
    global resolution
    global white_ball_initial_pos
    resolution = np.array([infoObject.current_w, infoObject.current_h])
    white_ball_initial_pos = (resolution + [table_margin + hole_radius, 0]) * [0.25, 0.5]

# window settings
fullscreen = False
# fullscreen resolution can only be known after initialising the screen
if not fullscreen:
    resolution = np.array([1000, 500])
window_caption = "Pool"
fps_limit = 60

# table settings
table_margin = 40
table_side_color = (200, 200, 0)
table_color = (0, 100, 0)
initial_point_color = (200, 200, 200)
initial_point_radius = 4
hole_radius = 22
middle_hole_offset = np.array([[-hole_radius * 2, hole_radius], [-hole_radius, 0],
                               [hole_radius, 0], [hole_radius * 2, hole_radius]])
side_hole_offset = np.array([
    [- 2 * math.cos(math.radians(45)) * hole_radius - hole_radius, hole_radius],
    [- math.cos(math.radians(45)) * hole_radius, -
    math.cos(math.radians(45)) * hole_radius],
    [math.cos(math.radians(45)) * hole_radius,
     math.cos(math.radians(45)) * hole_radius],
    [- hole_radius, 2 * math.cos(math.radians(45)) * hole_radius + hole_radius]
])

# cue settings
cue_color = (200, 100, 0)
cue_hit_power = 3
cue_length = 250
cue_thickness = 4
cue_max_displacement = 100
# safe displacement is the length the cue stick can be pulled before
# causing the ball to move
cue_safe_displacement = 1
aiming_line_length = 14

# ball settings
max_ball_num = 16
ball_radius = 14
ball_mass = 14
speed_angle_threshold = 0.09
visible_angle_threshold = 0.05
ball_colors = [
    (255, 255, 255),
    (  0, 200, 200),
    (  0, 133, 200),
    (  0,  67, 200),
    (  0,   0, 200),
    ( 67,   0, 200),
    (133,   0, 200),
    (200,   0, 200),
    (  0,   0,   0),
    (200,   0, 100),
    (200,   0,   0),
    (200,  67,   0),
    (200, 133,   0),
    (200, 200,   0),
    (100, 200,   0),
    (  0, 200, 100)
]
# where the balls will be placed at the start
# relative to screen resolution
ball_starting_place_ratio = [0.75, 0.5]
# in fullscreen mode the resolution is only available after initialising the screen
# and if the screen wasn't initialised the resolution variable won't exist
if 'resolution' in locals():
    white_ball_initial_pos = (resolution + [table_margin + hole_radius, 0]) * [0.25, 0.5]
ball_label_text_size = 10

# physics
# if the velocity of the ball is less then
# friction threshold then it is stopped
friction_threshold = 0.06
friction_coeff = 0.99
# 1 - perfectly elastic ball collisions
# 0 - perfectly inelastic collisions
ball_coeff_of_restitution = 0.9
table_coeff_of_restitution = 0.9

# menu
menu_text_color = (255, 255, 255)
menu_text_selected_color = (150, 150, 150)
menu_title_text = "Solo Pool"
menu_buttons = ["Play Solo Pool", "Exit"]
menu_margin = 20
menu_spacing = 10
menu_title_font_size = 40
menu_option_font_size = 20
exit_button = 2
play_game_button = 1

game_over_label_font_size = 40