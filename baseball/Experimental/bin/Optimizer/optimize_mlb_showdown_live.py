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
from optimizerTools import Locks, Exclusions, loadPlayers, getMaxLineupScore
from ProcessSlateStats import processSlateStats
import numpy as np

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
game_date = datetime.now().strftime("%Y-%m-%d")
datadir = f"{basedir}/../../data"
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/MLB/Showdown/{game_date}"
season = 2022
projdir = f"{basedir}/../../data/Projections/RealTime/{season}/Showdown/{game_date}"
### Read in Projections
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Showdown/{game_date}/{game_date}_Projections.csv"
)

slates = [
    file
    for file in os.listdir(
        f"{salary_dir}"
    )
    if ".csv" in file
]

n = 0
for file in slates:
    if len(file.split('_'))>2:
        print(n + 1, file)
        n += 1
if len(slates) == 1:
    slate = 1
else:
    slate = input("Select Slate by number (1,2,3, etc...) ")
slate = slates[int(slate) - 1]
slate_salaries=pd.read_csv(f'{salary_dir}/{slate}')
proj.drop('ID',axis=1,inplace=True)
proj=proj.merge(slate_salaries[['RotoName','team','position','Roster Position','ID','game_time',]],
                on=['RotoName','team','position','Roster Position','game_time'],
                how='left')
proj=proj[proj.ID.isna()==False]
# proj = processRankings(proj, game_date, Scaled=False)
processed = processSlateStats(proj,projdir,slate)


# Create optimizer
proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASEBALL)
proj["game_date"] = pd.to_datetime(proj.game_date)
players = loadPlayers(proj, "Projection", "Showdown")
optimizer.player_pool.load_players(players.to_list())
optimizer.set_min_salary_cap(48000)

# player = optimizer.player_pool.get_player_by_name(
#       "Kris Bryant", "CPT"
#   )  # using player name and position
# optimizer.player_pool.lock_player(player)
# player = optimizer.player_pool.get_player_by_name(
#       "Tyler Anderson","FLEX"
#   )  # using player name and position
# optimizer.player_pool.lock_player(player)
# optimizer.set_deviation(0,1)
for lineup in optimizer.optimize(n=10):
    print(lineup)
slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_Lineups.csv")
lineups = pd.read_csv(f"{datadir}/ExportedLineups/{slate}_Lineups.csv")
batter_db = pd.read_csv(f"{datadir}/game_logs/batterstatsDatabase.csv")
pitcher_db = pd.read_csv(f"{datadir}/game_logs/pitcherStatsDatabase.csv")
lineups["maxScore"] = lineups.T.apply(
    lambda x: getMaxLineupScore(x, proj, pitcher_db, batter_db,contest='Showdown')
)
lineups=lineups.sort_values(by="maxScore", ascending=False).head(20)
lineups.to_csv(
    f"{datadir}/ExportedLineups/{slate}_ShowdownLineups.csv", index=False
)
print(lineups[lineups.maxScore == lineups.maxScore.max()].T)
###

