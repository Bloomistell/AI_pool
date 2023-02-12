import itertools
import random

import zope.event

import config
import event
import physics


def resolve_all_collisions(balls, holes, table_sides):
    for ball_hole_combination in itertools.product(balls, holes):
        if physics.distance_less_equal(ball_hole_combination[0].pos, ball_hole_combination[1].pos, config.hole_radius):
            zope.event.notify(event.GameEvent("POTTED", ball_hole_combination[0]))

    for line_ball_combination in itertools.product(table_sides, balls):
        if physics.line_ball_collision_check(line_ball_combination[0], line_ball_combination[1]):
            physics.collide_line_ball(line_ball_combination[0], line_ball_combination[1])

    random.shuffle(balls)
    for ball_combination in itertools.combinations(balls, 2):
        if physics.ball_collision_check(ball_combination[0], ball_combination[1]):
            physics.collide_balls(ball_combination[0], ball_combination[1])
            zope.event.notify(event.GameEvent("COLLISION", ball_combination))


def check_if_ball_touches_balls(target_ball_pos, target_ball_number, balls):
    touches_other_balls = False
    for ball in balls:
        if target_ball_number != ball.number and \
                physics.distance_less_equal(ball.pos, target_ball_pos, config.ball_radius * 2):
            touches_other_balls = True
            break
    return touches_other_balls
