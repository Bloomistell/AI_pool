import itertools
import math
import random
from enum import Enum

import numpy as np
import pygame
import zope.event

import ball
import config
import cue
import event
import graphics
import table_sprites
from collisions import check_if_ball_touches_balls


class GameState:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(config.window_caption)
        event.set_allowed_events()
        zope.event.subscribers.append(self.game_event_handler)
        self.canvas = graphics.Canvas()
        self.fps_clock = pygame.time.Clock()

    def fps(self):
        return self.fps_clock.get_fps()

    def mark_one_frame(self):
        self.fps_clock.tick(config.fps_limit)

    def create_white_ball(self):
        self.white_ball = ball.BallSprite(0)
        ball_pos = config.white_ball_initial_pos
        while check_if_ball_touches_balls(ball_pos, 0, self.balls):
            ball_pos = [random.randint(int(config.table_margin + config.ball_radius + config.hole_radius),
                                       int(config.white_ball_initial_pos[0])),
                        random.randint(int(config.table_margin + config.ball_radius + config.hole_radius),
                                       int(config.resolution[1] - config.ball_radius - config.hole_radius))]
        self.white_ball.move_to(ball_pos)
        self.balls.add(self.white_ball)
        self.all_sprites.add(self.white_ball)

    def create_black_ball(self, initial_place, coord_shift):
        self.black_ball = ball.BallSprite(8)
        ball_pos = initial_place + coord_shift * [2, 0]
        self.black_ball.move_to(ball_pos)
        self.balls.add(self.black_ball)

    def game_event_handler(self, event):
        if event.type == "POTTED":
            if event.data.number == 0 or event.data.number == 8:
                self.game_over(False)
            else:
                self.table_coloring.update(self)
                self.balls.remove(event.data)
                self.all_sprites.remove(event.data)
                self.potted.append(event.data.number)

    def set_pool_balls(self, ball_number):
        assert 2 <= ball_number and ball_number <= config.max_ball_num, "Ball number is too low or too high, it must be between " + str(2) + " and " + str(config.max_ball_num) + "."

        counter = [0, 0]
        coord_shift = np.array([math.sin(math.radians(60)) * config.ball_radius *
                                2, -config.ball_radius])
        initial_place = config.ball_starting_place_ratio * config.resolution

        self.create_white_ball()
        self.create_black_ball(initial_place, coord_shift)
        # randomizes the sequence of balls on the table
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
                        ball_iteration = ball.BallSprite(ball_placement_sequence[ind])
                    else:
                        ball_iteration = ball.BallSprite(ball_placement_sequence[ind - 1])
                    ball_iteration.move_to(initial_place + coord_shift * counter)
                    self.balls.add(ball_iteration)
            if counter[1] == counter[0]:
                counter[0] += 1
                counter[1] = -counter[0]
            else:
                counter[1] += 2

        self.all_sprites.add(self.balls)

    def start_pool(self, ball_number = config.max_ball_num):
        self.reset_state()
        self.generate_table()
        self.set_pool_balls(ball_number)
        self.cue = cue.Cue(self.white_ball)
        self.all_sprites.add(self.cue)

    def reset_state(self):
        # game state variables
        self.turn_ended = True
        self.potted = []
        self.balls = pygame.sprite.Group()
        self.holes = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.OrderedUpdates()
        self.turn_number = 0
        self.is_game_over = False
        self.potting_8ball = False
        self.table_sides = []

    def is_behind_line_break(self):
        # 1st break should be made from behind the separation line on the table
        return self.turn_number == 0

    def redraw_all(self, update=True):
        self.all_sprites.clear(self.canvas.surface, self.canvas.background)
        self.all_sprites.draw(self.canvas.surface)
        self.all_sprites.update(self)
        if update:
            pygame.display.flip()
        self.mark_one_frame()

    def all_not_moving(self):
        return_value = True
        for ball in self.balls:
            if np.count_nonzero(ball.ball.velocity) > 0:
                return_value = False
                break
        return return_value

    def generate_table(self):
        table_side_points = np.empty((1, 2))
        # holes_x and holes_y holds the possible xs and ys of the table holes
        # with a position ID in the second tuple field
        # so the top left hole has id 1,1
        holes_x = [(config.table_margin, 1), (config.resolution[0] /
                                              2, 2), (config.resolution[0] - config.table_margin, 3)]
        holes_y = [(config.table_margin, 1),
                   (config.resolution[1] - config.table_margin, 2)]
        # next three lines are a hack to make and arrange the hole coordinates
        # in the correct sequence
        all_hole_positions = np.array(
            list(itertools.product(holes_y, holes_x)))
        all_hole_positions = np.fliplr(all_hole_positions)
        all_hole_positions = np.vstack(
            (all_hole_positions[:3], np.flipud(all_hole_positions[3:])))
        for hole_pos in all_hole_positions:
            self.holes.add(table_sprites.Hole(hole_pos[0][0], hole_pos[1][0]))
            # this will generate the diagonal, vertical and horizontal table
            # pieces which will reflect the ball when it hits the table sides
            #
            # they are generated using 4x2 offset matrices (4 2d points around the hole)
            # with the first point in the matrix is the starting point and the
            # last point is the ending point, these 4x2 matrices are
            # concatenated together
            #
            # the martices must be flipped using numpy.flipud()
            # after reflecting them using 2x1 reflection matrices, otherwise
            # starting and ending points would be reversed
            if hole_pos[0][1] == 2:
                # hole_pos[0,1]=2 means x coordinate ID is 2 which means this
                # hole is in the middle
                offset = config.middle_hole_offset
            else:
                offset = config.side_hole_offset
            if hole_pos[1][1] == 2:
                offset = np.flipud(offset) * [1, -1]
            if hole_pos[0][1] == 1:
                offset = np.flipud(offset) * [-1, 1]
            table_side_points = np.append(
                table_side_points, [hole_pos[0][0], hole_pos[1][0]] + offset, axis=0)
        # deletes the 1st point in array (leftover form np.empty)
        table_side_points = np.delete(table_side_points, 0, 0)
        for num, point in enumerate(table_side_points[:-1]):
            # this will skip lines inside the circle
            if num % 4 != 1:
                self.table_sides.append(table_sprites.TableSide(
                    [point, table_side_points[num + 1]]))
        self.table_sides.append(table_sprites.TableSide(
            [table_side_points[-1], table_side_points[0]]))
        self.table_coloring = table_sprites.TableColoring(
            config.resolution, config.table_side_color, table_side_points)
        self.all_sprites.add(self.table_coloring)
        self.all_sprites.add(self.holes)
        graphics.add_initial_point(self.canvas)

    def game_over(self, won):
        font = config.get_default_font(config.game_over_label_font_size)
        if won:
            text = "WON!"
        else:
            text = "LOOSE :("
        rendered_text = font.render(text, False, (255, 255, 255))
        self.canvas.surface.blit(rendered_text, (config.resolution - font.size(text)) / 2)
        pygame.display.flip()
        pygame.event.clear()
        paused = True
        while paused:
            event = pygame.event.wait()
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                paused = False
        self.is_game_over = True

    def turn_over(self):
        if not self.turn_ended:
            self.turn_ended = True
            self.turn_number += 1

    def check_potted(self):
        if 0 in self.potted:
            self.game_over(False)
        if 8 in self.potted:
            if self.potting_8ball:
                self.game_over(True)
            else:
                self.game_over(False)

    def check_remaining(self):
        # a check if all striped or solid balls were potted
        remaining = False
        for remaining_ball in self.balls:
            remaining = remaining_ball.number != 0 and remaining_ball.number != 8
        ball_remaining = remaining

        # decides if on of the players (or both) should be potting 8ball
        self.potting_8ball = not ball_remaining

    def check_pool_rules(self):
        self.check_remaining()
        self.check_potted()
        self.turn_over()
        self.on_next_hit()

    def on_next_hit(self):
        self.turn_ended = False
        self.potted = []