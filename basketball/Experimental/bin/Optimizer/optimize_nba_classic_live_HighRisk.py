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
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
game_date = datetime.now().strftime("%Y-%m-%d")
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
proj = proj[(proj.Projection.isna() == False) & (proj.Salary.isna() == False)]
proj = processRankings(proj, game_date, powerplay=False)


# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.HOCKEY)

players = loadPlayers(proj, "Projection", "Classic")
optimizer.player_pool.load_players(players.to_list())
# optimizer.set_min_salary_cap(49500)
optimizer.restrict_positions_for_opposing_team(["G"], ["LW", "RW", "D", "C"])
team1 = proj[proj.moneyline < 0].sample(1).team.values[0]
team2 = team1
while team1 == team2:
    team2 = proj.sample(1).team.values[0]
stack1 = np.random.choice(range(4, 7))
stack2 = 8 - stack1
optimizer.add_stack(TeamStack(stack1, for_teams=[team1]))
optimizer.add_stack(TeamStack(stack2, for_teams=[team2]))
optimizer.set_total_teams(max_teams=3)
Exclude = ["Patrice Bergeron"]
# Exclusions(Exclude, optimizer)
# optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(1.0, 10.0))
# optimizer.player_pool.lock_player("Kevin Shattenkirk")
# optimizer.set_total_teams(max_teams=3)# Execute Optimizer
for lineup in optimizer.optimize(n=10, randomness=True):
    print(lineup)
slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_HighRiskLineups.csv")
