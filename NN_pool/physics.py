import numpy as np

import config


def point_distance(p1, p2):
    dist_diff = p1 - p2
    return np.hypot(*dist_diff)


def distance_less_equal(p1, p2, dist):
    dist_diff = p1 - p2
    return (dist_diff[0] ** 2 + dist_diff[1] ** 2) <= dist ** 2


def ball_collision_check(ball1, ball2):
    return distance_less_equal(ball1.pos, ball2.pos, 2 * config.ball_radius) and \
           np.count_nonzero(np.concatenate((ball1.velocity, ball2.velocity))) > 0 and \
           np.dot(ball2.pos - ball1.pos, ball1.velocity - ball2.velocity) > 0


def collide_balls(ball1, ball2):
    point_diff = ball2.pos - ball1.pos
    dist = point_distance(ball1.pos, ball2.pos)
    collision = point_diff / dist
    ball1_dot = np.dot(ball1.velocity, collision)
    ball2_dot = np.dot(ball2.velocity, collision)
    ball1.velocity += (ball2_dot - ball1_dot) * collision * 0.5*(1+config.ball_coeff_of_restitution)
    ball2.velocity += (ball1_dot - ball2_dot) * collision * 0.5*(1+config.ball_coeff_of_restitution)


def line_ball_collision_check(line, ball):
    if distance_less_equal(line.middle, ball.pos, line.length / 2 + config.ball_radius):
        displacement_to_ball = ball.pos - line.line[0]
        displacement_to_second_point = line.line[1] - line.line[0]
        normalised_point_diff_vector = displacement_to_second_point / \
                                       np.hypot(*(displacement_to_second_point))
        projected_distance = np.dot(normalised_point_diff_vector, displacement_to_ball)
        closest_line_point = projected_distance * normalised_point_diff_vector
        perpendicular_vector = np.array(
            [-normalised_point_diff_vector[1], normalised_point_diff_vector[0]])
        return -config.ball_radius / 3 <= projected_distance <= \
               np.hypot(*(displacement_to_second_point)) + config.ball_radius / 3 and \
               np.hypot(*(closest_line_point - ball.pos + line.line[0])) <= \
               config.ball_radius and np.dot(
            perpendicular_vector, ball.velocity) <= 0


def collide_line_ball(line, ball):
    displacement_to_second_point = line.line[1] - line.line[0]
    normalised_point_diff_vector = displacement_to_second_point / \
                                   np.hypot(*(displacement_to_second_point))
    perpendicular_vector = np.array(
        [-normalised_point_diff_vector[1], normalised_point_diff_vector[0]])
    ball.velocity -= 2 * np.dot(perpendicular_vector,ball.velocity) * \
                     perpendicular_vector * 0.5*(1+config.table_coeff_of_restitution)
