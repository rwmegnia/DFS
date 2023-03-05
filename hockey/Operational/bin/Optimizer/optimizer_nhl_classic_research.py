#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:05:01 2022

@author: robertmegnia
"""

# -*- coding: utf-8 -*-
import pandas as pd
from pydfs_lineup_optimizer import Site, Sport, get_optimizer, PositionsStack, TeamStack

# from pydfs_lineup_optimizer.stacks import GameStack
from pydfs_lineup_optimizer.fantasy_points_strategy import RandomFantasyPointsStrategy
from pydfs_lineup_optimizer.exposure_strategy import AfterEachExposureStrategy
import os
import numpy as np
import warnings
from datetime import datetime
from ProcessRankings import processRankings
from optimizerTools import Locks, Exclusions, loadPlayers, print_lineups

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
game_date = "2022-03-21"
contestType = "Classic"
season = 2021
### Prompt User to Select a Slate of Projections
projection_files = os.listdir(
    f"{datadir}/Projections/RealTime/{season}/{contestType}/{game_date}"
)
n = 0
for file in projection_files:
    print(n + 1, file)
if len(projection_files) == 1:
    slate = 1
else:
    slate = input("Select Slate by number (1,2,3, etc...) ")
slate = projection_files[int(slate) - 1]
# Read in Projections
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/{contestType}/{game_date}/{slate}"
)
# Use these Settings for research projection files.

# proj = pd.read_csv(f"{datadir}/Projections/{season}/{game_date}_Projections.csv")
# proj["Salary"] = proj["Salary_y"]
# proj["game_id"] = "derp"
# proj["ownership_proj"] = 0
# proj["home_team"] = "home"
# proj["away_team"] = "away"
# proj["ID"] = proj.player_id
# proj["line"] = proj.line_x
# proj["RG_projection"] = proj.TD_Proj
#
#### Settings for research projectiosn files ends here
proj = proj[(proj.Projection.isna() == False) & (proj.Salary.isna() == False)]
proj["game_date"] = proj.game_date.astype(str)

proj = processRankings(proj, game_date, powerplay=False)

# proj["Projection"] = proj[["RG_projection", "RG_projection", "Projection"]].mean(axis=1)
# proj=proj[proj.line<4]

# Create optimizer
if contestType == "Showdown":
    proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
    proj.loc[proj["Roster Position"] == "CPT", "DKPts"] *= 1.5
    optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.HOCKEY)
else:
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.HOCKEY)
players = loadPlayers(proj, "RG_projection", contestType)
optimizer.player_pool.load_players(players.to_list())
# optimizer.set_min_salary_cap(49700)
optimizer.add_stack(PositionsStack([["G"], ("LW", "C", "RW", "D")]))
# optimizer.add_stack(PositionsStack(['LW','C','RW']))

optimizer.restrict_positions_for_opposing_team(["G"], ["LW", "RW", "D", "C"])
# optimizer.set_projected_ownership(200,220)
team1, team2 = (
    proj.groupby("team")
    .agg(
        {
            "Projection": np.sum,
            "moneyline": np.mean,
            "proj_team_score": np.mean,
            "opp": "first",
        }
    )
    .sort_values(by="moneyline")
    .index[0:2]
)
# optimizer.add_stack(
#     TeamStack(4, for_positions=["C", "LW", "RW", "D"], for_teams=[team1])
# )
# optimizer.add_stack(
#     TeamStack(3, for_positions=["C", "LW", "RW"], for_teams=[team2])
# )
# optimizer.add_stack(TeamStack(1,for_positions=['C','LW','RW']))
Exclude = ["Sergei Bobrovsky"]
# Exclusions(Exclude, optimizer)
# optimizer.player_pool.exclude_teams(['SEA'])
# optimizer.player_pool.lock_player('David Pastrnak')
# optimizer.set_total_teams(max_teams=3)
# Execute Optimizer
LineupResults = []
AllLineupResults = []
for lineup in optimizer.optimize(n=10):
    print_lineups(lineup, proj)
    LineupResults.append(lineup.actual_fantasy_points_per_game_projection)
    AllLineupResults.append(lineup.actual_fantasy_points_per_game_projection)

print(np.mean(LineupResults))
print(np.max(LineupResults))
print(np.mean(LineupResults))
print(np.max(LineupResults))
