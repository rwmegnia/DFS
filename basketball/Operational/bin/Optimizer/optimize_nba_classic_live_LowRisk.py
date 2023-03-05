# -*- coding: utf-8 -*-
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
datadir = f"{basedir}/../../data"
game_date = datetime.now().strftime("%Y-%m-%d")
# game_date='2022-04-01'
season = 2021

### Prompt User to Select a Slate of Projections
projection_files = os.listdir(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}"
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
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{slate}"
)
proj = processRankings(proj, Scaled=True)
proj = proj[(proj.Projection.isna() == False) & (proj.Salary.isna() == False)]


# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)
players = loadPlayers(proj, "Projection", "Classic")
optimizer.player_pool.load_players(players.to_list())


Exclude = []
Exclusions(Exclude, optimizer)
optimizer.player_pool.exclude_teams(['OKC','CHI'])
# optimizer.player_pool.lock_player("Kevin Shattenkirk")
# Execute Optimizer
for lineup in optimizer.optimize(n=10):
    print(lineup)
slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_LowRiskLineups.csv")
