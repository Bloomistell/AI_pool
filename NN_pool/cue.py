import math

import numpy as np

import config


class Cue():
    def __init__(self, target):
        self.target_ball = target

    def ball_hit(self, angle, displacement):
        displacement = max(min(displacement, config.cue_max_displacement), config.ball_radius)
        new_velocity = -(displacement - config.ball_radius - config.cue_safe_displacement) * \
                       config.cue_hit_power * np.array([math.sin(angle), math.cos(angle)])
        self.target_ball.apply_force(new_velocity)
