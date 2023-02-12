import numpy as np
import config


class Ball():
    def __init__(self, ball_number):
        self.number = ball_number
        self.pos = np.zeros(2, dtype=float)
        self.velocity = np.zeros(2, dtype=float)

    def apply_force(self, force, time=1):
        # f = ma, v = u + at -> v = u + (f/m)*t
        self.velocity += (force / config.ball_mass) * time

    def move_to(self, pos):
        self.pos = np.array(pos, dtype=float)

    def update(self):
        if np.hypot(*self.velocity) != 0:
            self.velocity *= config.friction_coeff
            self.pos += self.velocity

            if np.hypot(*self.velocity) < config.friction_threshold:
                self.velocity = np.zeros(2)