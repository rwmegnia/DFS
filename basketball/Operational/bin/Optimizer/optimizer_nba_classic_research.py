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
game_date = "2022-04-05"
contestType = "Classic"
season = 2021


def mse(a, b):
    x = round(np.mean((a - b) ** 2), 2)
    return x


def mae(a, b):
    rms = round(np.mean(np.abs(a - b)), 2)
    return rms


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
proj.drop_duplicates(inplace=True)
#### Settings for research projectiosn files ends here
proj = proj[(proj.Projection.isna() == False) & (proj.Salary.isna() == False)]
proj["game_date"] = proj.game_date.astype(str)

proj = processRankings(proj,Scaled=False)
proj.loc[proj.RG_projection==0,'Projection']=0
proj=proj[proj.dkpts>0]
# Create optimizer
if contestType == "Showdown":
    proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
    proj.loc[proj["Roster Position"] == "CPT", "DKPts"] *= 1.5
    optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASKETBALL)
else:
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)
players = loadPlayers(proj, "Projection", contestType)
optimizer.player_pool.load_players(players.to_list())
# optimizer.set_min_salary_cap(49700)

# optimizer.set_projected_ownership(200,220)
Exclude = []
Exclusions(Exclude, optimizer)
# optimizer.player_pool.exclude('David Pastrnak')
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
