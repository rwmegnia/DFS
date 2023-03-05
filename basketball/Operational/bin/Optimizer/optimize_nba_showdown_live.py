#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:40:06 2022

@author: robertmegnia
"""
import pandas as pd
from pydfs_lineup_optimizer import Site, Sport, get_optimizer, PositionsStack, TeamStack
from pydfs_lineup_optimizer.fantasy_points_strategy import RandomFantasyPointsStrategy
import os
import warnings
from datetime import datetime
from ProcessRankings import processRankings
from optimizerTools import Locks, Exclusions, loadPlayers
import numpy as np

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
game_date = datetime.now().strftime("%Y-%m-%d")
season = 2021
### Prompt User to Select a Slate of Projections
projection_files = os.listdir(
    f"{datadir}/Projections/RealTime/{season}/Showdown/{game_date}"
)
n = 0
for file in projection_files:
    print(n + 1, file)
    n += 1
if len(projection_files) == 1:
    slate = 1
else:
    slate = input("Select Slate by number (1,2,3, etc...) ")
slate = projection_files[int(slate) - 1]
### Read in Projections
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Showdown/{game_date}/{slate}"
)
proj = proj[(proj.Projection.isna() == False) & (proj.Salary.isna() == False)]
proj = processRankings(proj, game_date, powerplay=False)


# Create optimizer
proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.HOCKEY)
players = loadPlayers(proj, "Projection", "Showdown")
optimizer.player_pool.load_players(players.to_list())


# player = optimizer.player_pool.get_player_by_name(
#     "Cale Makar", "FLEX"
# )  # using player name and position
# optimizer.player_pool.lock_player(player)
for lineup in optimizer.optimize(n=100, randomness=True):
    print(lineup)
slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_Lineups.csv")
