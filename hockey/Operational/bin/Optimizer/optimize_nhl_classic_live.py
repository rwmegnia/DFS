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
proj = processRankings(proj, game_date, powerplay=True)
proj = proj[proj.line < 4]
proj["Projection"] = proj[["Ceiling", "Projection", "RG_projection"]].mean(axis=1)

# Create optimizer
# Create optimizerti
if contestType == "Showdown":
    proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
    optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.HOCKEY)
else:
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.HOCKEY)

players = loadPlayers(proj, "Projection", contestType)
optimizer.player_pool.load_players(players.to_list())
# optimizer.set_min_salary_scap(49800)
optimizer.add_stack(PositionsStack([["G"], ("LW", "C", "RW", "D")]))
optimizer.set_spacing_for_positions(["LW", "RW", "C", "D"], 2)
optimizer.restrict_positions_for_opposing_team(["G"], ["LW", "RW", "D", "C"])
# optimizer.set_players_with_same_position({'LW':2,'RW':2})
# optimizer.set_projected_ownership(80,100)
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
#     TeamStack(4, for_positions=["C", "LW", "RW", "D"], for_teams=["NSH"])
# )
# optimizer.add_stack(
#     TeamStack(3, for_positions=["C", "LW", "RW", "D"], for_teams=["BOS"])
# )

# Exclude=['Semyon Varlamov']
# Exclusions(Exclude,optimizer)
# optimizer.player_pool.exclude_teams(["SJS"])
# optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(1.0, 10.0))
# player = optimizer.player_pool.get_player_by_name(
#     "Cale Makar", "FLEX"
# )  # using player name and position
# optimizer.player_pool.lock_player(player)
# optimizer.player_pool.lock_player("Kevin Shattenkirk")
# optimizer.player_pool.lock_player("Jeremy Swayman")
# optimizer.player_pool.lock_player("Roman Josi")
# optimizer.player_pool.lock_player("Connor McDavid")
# optimizer.player_pool.lock_player("Charlie Coyle")
# optimizer.player_pool.lock_player("David Pastrnak")
# optimizer.player_pool.lock_player("Charlie Mcavoy")
# optimizer.set_total_teams(max_teams=3)# Execute Optimizer
for lineup in optimizer.optimize(n=10):
    print(lineup)
slate = slate.split("_Projections.csv")[0]
optimizer.export(f"{datadir}/ExportedLineups/{slate}_Lineups.csv")
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_Lineups.csv")
