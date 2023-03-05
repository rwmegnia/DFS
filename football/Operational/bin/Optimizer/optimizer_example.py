# -*- coding: utf-8 -*-
"""

Optimizer used for verification.

"""
import pandas as pd
import numpy as np
from pydfs_lineup_optimizer import (
    Site,
    Sport,
    get_optimizer,
    PositionsStack,
    TeamStack,
    CSVLineupExporter,
    AfterEachExposureStrategy,
)
from pydfs_lineup_optimizer.stacks import GameStack
from pydfs_lineup_optimizer.fantasy_points_strategy import *
from pydfs_lineup_optimizer.solvers.pulp_solver import PuLPSolver
from pydfs_lineup_optimizer.fantasy_points_strategy import (
    ProgressiveFantasyPointsStrategy,
)
from pydfs_lineup_optimizer import Player
from pydfs_lineup_optimizer.player import GameInfo, Player
from pydfs_lineup_optimizer.tz import get_timezone
from pytz import timezone
from OptimizerTools import *
import os
import warnings
from datetime import datetime

# Read in Projections data frame and create player pool
proj = pd.read_csv(
            f"PathToProjections"
        )

# Create list of pydfs linup optmizer player opjects
players = proj.apply(
    lambda x: Player(
        player_id=x['ID_Columns'],
        first_name=x.full_name.split(" ")[0], # Add first name column
        last_name=" ".join(x.full_name.split(" ")[1::]), # last name column
        positions=[x.position],# position column
        team=x.team, # team coumn
        salary=x.salary, # salary column
        fppg=x.Projection, # projection column
        projected_ownership=x.AvgOwnership/100, #ownerhsip column
        ),
    axis=1,
        )
        # Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
optimizer.player_pool.load_players(players.to_list())


# Apply rules
optimizer.set_players_with_same_position({"WR": 1})
optimizer.add_stack(PositionsStack(["RB"],for_teams=teams))
optimizer.add_stack(PositionsStack(["RB"],for_teams=teams))
optimizer.restrict_positions_for_opposing_team(
            ["DST"], ["QB", "RB", "WR", "TE"]
        )
Exclude=['Rondale Moore','KJ Hamler','Kadarius Toney','Jakobi Meyers']
# Exclusions(Exclude,optimizer)
# optimizer.player_pool.lock_player('Adam Thielen')
# optimizer.player_pool.lock_player('Deebo Samuel')
# optimizer.player_pool.lock_player('Justin Jefferson')
# optimizer.player_pool.lock_player('Aaron Jones')
# optimizer.player_pool.lock_player('Trey Lance')

# optimizer.player_pool.lock_player('Keenan Allen')
# optimizer.player_pool.lock_player('Davante Adams')
# optimizer.set_total_teams(3)
# optimizer.set_max_repeating_players(6)
# optimizer.force_positions_for_opposing_team(("QB", "WR"))
# optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0,1))
# optimizer.set_min_salary_cap(49500)
# Execute Optimizer
LineupResults = []
for lineup in optimizer.optimize(n=10,):
# for lineup in optimizer.optimize(n=10):
    print(lineup)
optimizer.print_statistic()