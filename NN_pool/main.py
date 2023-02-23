import sys

import collisions
import event
import gamestate

if len(sys.argv) <= 1:
    game = gamestate.GameState()
else:
    game = gamestate.GameState(int(sys.argv[1]))

game = gamestate.GameState(2,poolEnv = False0)
while not game.is_game_over:
    collisions.resolve_all_collisions(game.balls, game.holes, game.table_sides)
    game.update_balls()

    if game.balls_not_moving():
        game.check_pool_rules()
