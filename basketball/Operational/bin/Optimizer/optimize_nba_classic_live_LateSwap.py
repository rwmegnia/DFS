# -*- coding: utf-8 -*-
import pandas as pd
from pydfs_lineup_optimizer import Site, Sport, get_optimizer, PositionsStack, TeamStack
from pydfs_lineup_optimizer.fantasy_points_strategy import RandomFantasyPointsStrategy
import os
import warnings
from datetime import datetime
from ProcessRankings import processRankings
from optimizerTools import Locks, Exclusions, loadPlayers, mergeSwapProjectedPoints
import numpy as np

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
game_date = datetime.now().strftime("%Y-%m-%d")
# game_date = "2022-03-16"
contestType = "Classic"
season = 2021
### Prompt User to Select a Slate of Projections
projection_files = os.listdir(
    f"{datadir}/Projections/RealTime/{season}/{contestType}/{game_date}"
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
    f"{datadir}/Projections/RealTime/{season}/{contestType}/{game_date}/{slate}"
)
proj = proj[(proj.Projection.isna() == False) & (proj.Salary.isna() == False)]
#proj = processRankings(proj)
proj.loc[proj.RG_projection==0,'Projection']=0

# Create optimizerti
if contestType == "Showdown":
    proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
    optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASKETBALL)
else:
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)


optimizer.load_players_from_csv("/Users/robertmegnia/Downloads/DKEntries.csv")
players = [
    mergeSwapProjectedPoints(p, proj)
    for p in optimizer.player_pool.all_players
    if mergeSwapProjectedPoints(p, proj) is not None
]
optimizer.player_pool.load_players(players)
lineups = optimizer.load_lineups_from_csv("/Users/robertmegnia/Downloads/DKEntries.csv")
# optimizer.set_min_salary_cap(49500)
# Exclude = ["Troy Terry"]
# Exclusions(Exclude, optimizer)

for lineup in optimizer.optimize_lineups(lineups):
    print(lineup)
slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_LineupsLateSwap.csv")
