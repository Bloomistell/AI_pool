import itertools
import math
import random

import numpy as np
import zope.event

import ball
import config
import cue
import table_sprites
from collisions import check_if_ball_touches_balls


class GameState:
    def __init__(self, ball_number = config.max_ball_num):
        zope.event.subscribers.append(self.game_event_handler)
        self.reset_state()
        self.set_pool_balls(ball_number)
        self.generate_table()
        self.cue = cue.Cue(self.white_ball)

    def game_event_handler(self, event):
        if event.type == "POTTED":
            if event.data.number == 0 or event.data.number == 8:
                self.game_over(False)
            else:
                self.balls.remove(event.data)
                self.potted.append(event.data.number)
    
    def reset_state(self):
        self.potted = []
        self.balls = []
        self.holes = []
        self.turn_number = 0
        self.is_game_over = False
        self.potting_8ball = False
        self.table_sides = []
    
    def create_white_ball(self):
        self.white_ball = ball.Ball(0)
        ball_pos = config.white_ball_initial_pos
        while check_if_ball_touches_balls(ball_pos, 0, self.balls):
            ball_pos = [random.randint(int(config.table_margin + config.ball_radius + config.hole_radius),
                                       int(config.white_ball_initial_pos[0])),
                        random.randint(int(config.table_margin + config.ball_radius + config.hole_radius),
                                       int(config.resolution[1] - config.ball_radius - config.hole_radius))]
        self.white_ball.move_to(ball_pos)
        self.balls.append(self.white_ball)

    def create_black_ball(self, initial_place, coord_shift):
        self.black_ball = ball.Ball(8)
        ball_pos = initial_place + coord_shift * [2, 0]
        self.black_ball.move_to(ball_pos)
        self.balls.append(self.black_ball)

    def set_pool_balls(self, ball_number):
        assert 2 <= ball_number and ball_number <= config.max_ball_num, "Ball number is too low or too high, it must be between " + str(2) + " and " + str(config.max_ball_num) + "."

        counter = [0, 0]
        coord_shift = np.array([math.sin(math.radians(60)) * config.ball_radius * 2, -config.ball_radius])
        initial_place = config.ball_starting_place_ratio * config.resolution

        self.create_white_ball()
        self.create_black_ball(initial_place, coord_shift)
        
        ball_placement_sequence = list(range(1, ball_number))
        if 8 in ball_placement_sequence:
            ball_placement_sequence.remove(8)
        else: 
            ball_placement_sequence.pop()
        random.shuffle(ball_placement_sequence)

        for ind in range(len(ball_placement_sequence) + 1):
            if ind != 4:
                if ind > 4 or ind < len(ball_placement_sequence):
                    if ind < 4:
                        ball_iteration = ball.Ball(ball_placement_sequence[ind])
                    else:
                        ball_iteration = ball.Ball(ball_placement_sequence[ind - 1])
                    ball_iteration.move_to(initial_place + coord_shift * counter)
                    self.balls.append(ball_iteration)
            if counter[1] == counter[0]:
                counter[0] += 1
                counter[1] = -counter[0]
            else:
                counter[1] += 2
    
    def generate_table(self):
        table_side_points = np.empty((1, 2))
        holes_x = [(config.table_margin, 1), (config.resolution[0] / 2, 2), (config.resolution[0] - config.table_margin, 3)]
        holes_y = [(config.table_margin, 1), (config.resolution[1] - config.table_margin, 2)]
        all_hole_positions = np.array(list(itertools.product(holes_y, holes_x)))
        all_hole_positions = np.fliplr(all_hole_positions)
        all_hole_positions = np.vstack(
            (all_hole_positions[:3], np.flipud(all_hole_positions[3:])))
        for hole_pos in all_hole_positions:
            self.holes.append(table_sprites.Hole(hole_pos[0][0], hole_pos[1][0]))
            if hole_pos[0][1] == 2:
                offset = config.middle_hole_offset
            else:
                offset = config.side_hole_offset
            if hole_pos[1][1] == 2:
                offset = np.flipud(offset) * [1, -1]
            if hole_pos[0][1] == 1:
                offset = np.flipud(offset) * [-1, 1]
            table_side_points = np.append(
                table_side_points, [hole_pos[0][0], hole_pos[1][0]] + offset, axis=0)
        table_side_points = np.delete(table_side_points, 0, 0)
        for num, point in enumerate(table_side_points[:-1]):
            if num % 4 != 1:
                self.table_sides.append(table_sprites.TableSide([point, table_side_points[num + 1]]))
        self.table_sides.append(table_sprites.TableSide([table_side_points[-1], table_side_points[0]]))

    def update_balls(self):
        for ball in self.balls:
            ball.update()

    def balls_not_moving(self):
        return_value = True
        for ball in self.balls:
            if np.count_nonzero(ball.velocity) > 0:
                return_value = False
                break
        return return_value

    def game_over(self, won):
        self.won = won
        self.is_game_over = True

    def check_pool_rules(self):
        self.check_remaining()
        self.check_potted()
        self.turn_over()
        self.next_turn()

    def check_remaining(self):
        remaining = False
        for remaining_ball in self.balls:
            remaining = remaining_ball.number != 0 and remaining_ball.number != 8
        ball_remaining = remaining
        self.potting_8ball = not ball_remaining

    def check_potted(self):
        if 0 in self.potted:
            self.game_over(False)
        if 8 in self.potted:
            if self.potting_8ball:
                self.game_over(True)
            else:
                self.game_over(False)
    
    def turn_over(self):
        self.turn_number += 1
        print("State :")
        for ball in self.balls:
            print(ball.number, ": ", ball.pos)


    def next_turn(self):
        print("Tour :", self.turn_number)
        print("Nombre de balles rentrées :", len(self.potted))
        angle = int(input("Angle de tir (0 à 360 degrés) ? "))
        displacement = int(input("Puissance de tir (0 à " + str(config.cue_max_displacement) + ") ? "))
        self.cue.ball_hit(angle, displacement)
        self.potted = []
