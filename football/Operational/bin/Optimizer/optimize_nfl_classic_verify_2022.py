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
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"

# from getConsensusRank import getConsensusRanking

# Read in Projections data frame and create player pool
season = 2022
week = 9
AllLineupResults = []
method = "Projection"
pre_bc_frames = []
post_bc_frames = []
team_frames = []
leverage_plays = []
total_cash = 0
total_contests = 0
proj = pd.read_csv(
    f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections_verified.csv"
)
proj.rename({'DKPts_x':'DKPts'},axis=1,inplace=True)
proj['Projection']=proj[['Stochastic','ML','TopDown',"DC_proj","Median"]].mean(axis=1)
proj=proj.groupby('gsis_id').first()
proj.reset_index(inplace=True)
# game_date=proj.game_time.min()[0:10]
game_date = "2022-11-03"
salary_dir = (
    f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NFL/Classic/{game_date}"
)

slates = [
    file
    for file in os.listdir(f"{salary_dir}")
    if (".csv" in file) & (f"{game_date}_salaries.csv" not in file)
]

n = 0
for file in slates:
    if len(file.split("_")) > 2:
        print(n + 1, file)
        n += 1
if len(slates) == 1:
    slate = 1
else:
    slate = input("Select Slate by number (1,2,3, etc...) ")
slate = slates[int(slate) - 1]
slate_salaries = pd.read_csv(f"{salary_dir}/{slate}")
slate_salaries.loc[
    slate_salaries.position == "DST", "RotoName"
] = slate_salaries.loc[slate_salaries.position == "DST", "team"]
proj.drop(["ID", "salary"], axis=1, inplace=True, errors="ignore")
proj = proj.merge(
    slate_salaries[["RotoName", "salary", "ID", "Roster Position"]],
    on=["RotoName", "Roster Position"],
    how="left",
)
proj = filterProjections(proj, slate_salaries)
proj["Ownership"] = proj.AvgOwnership / 100
#
# Get Individual Game Information
game_info = {}
game_ids = proj[proj.game_id != 0].groupby("game_id", as_index=False).first()
for row in game_ids.iterrows():
    row = row[1]
    game_info[row.game_id] = GameInfo(
        row.home_team, row.away_team, row.game_date
    )
# Load Players
players = proj.apply(
    lambda x: Player(
        player_id=x.ID,
        first_name=x.full_name.split(" ")[0],
        last_name=" ".join(x.full_name.split(" ")[1::]),
        positions=[x.position],
        team=x.team,
        salary=x.salary,
        # min_deviation=x.min_deviation,
        # max_deviation=x.max_deviation,
        fppg=x[method],
        projected_ownership=x.Ownership,
        game_info=game_info[x.game_id],
    ),
    axis=1,
)
# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
optimizer.player_pool.load_players(players.to_list())
# optimizer.set_min_salary_cap(50000)
# optimizer.player_pool.exclude_teams(['PIT','HOU','LA','CAR','DEN','IND'])

# optimizer.set_players_with_same_position({"RB": 1})

pass_teams = getPassingStackTeams(proj)
rb_teams = getRunningBackTeams(proj)


# # # Exclude or Lock Players
# Exclude = ["Breece Hall"]
# Exclusions(Exclude,optimizer)
Include = ["Justin Fields","Justin Jefferson"]
Locks(Include,optimizer)

###


# optimizer.set_projected_ownership(11,14)
# optimizer.add_stack(PositionsStack(["QB", "WR"]))
# optimizer.add_stack(PositionsStack(["RB"], for_teams=rb_teams))
# optimizer.add_stack(PositionsStack(["RB"],for_teams=rb_teams))

# optimizer.set_projected_ownership(0.10,0.11)
# optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0.5,1.5))
optimizer.restrict_positions_for_opposing_team(
    ["DST"], ["QB", "RB", "WR", "TE"]
)
# optimizer.force_positions_for_opposing_team(("QB", "RB"))
# Execute Optimizer
LineupResults = []
for lineup in optimizer.optimize(
    n=10
):
    lineup_frame = print_lineups(lineup, proj)
    LineupResults.append(lineup.actual_fantasy_points_per_game_projection)
    AllLineupResults.append(lineup.actual_fantasy_points_per_game_projection)

print(f"Average Points: {np.mean(LineupResults)}")
print(f"Highest Score: {np.max(LineupResults)}")
print(np.mean(AllLineupResults))
print(np.max(AllLineupResults))
# proj = proj[proj.RG_projection != 0]
# proj=proj[proj[['RG_projection',
#                 'FP_Proj',
#                 'FP_Proj2',
#                 'GI',
#                 'NicksAgg']].isna().any(axis=1)==False]
# proj=proj[proj.RosterPercent>0]
# proj.rename({'RG_projection':'RotoGrinders',
#               'FP_Proj':'FantasyPros',
#               'FP_Proj2':'FP_Expert_Cons',
#               'GI':'GridIron',
#               'NicksAgg':'Nick',
#               'wProjection':'Rob'},axis=1,inplace=True)
# for method in ['RotoGrinders','FantasyPros','FP_Expert_Cons','GridIron','Nick','Rob']:
#     proj[f'{method}_mae']=proj.apply(lambda x: mae(x.DKPts,x[method]),axis=1)

# maes=proj.mean()[['RotoGrinders_mae','FantasyPros_mae','FP_Expert_Cons_mae','GridIron_mae','Nick_mae','Rob_mae']]
# maes.sort_values(inplace=True)
# maes=maes.round(2)
# fig,ax=plt.subplots()
# maes.plot.bar(ax=ax)
# ax.set_xticks(range(0,6),rotation=45)
# ax.set_xticklabels(maes.index,rotation=360)
# ax.set_ylim(4.5,5.35,0.5)
# ax.set_yticks(np.arange(4.5,5.35,.05))
# ax.annotate(maes.Nick_mae,(-.05,maes.Nick_mae+.05))
# ax.annotate(maes.RotoGrinders_mae,(0.95,maes.RotoGrinders_mae+.05))
# ax.annotate(maes.Rob_mae,(1.95,maes.Rob_mae+.05))
# ax.annotate(maes.FP_Expert_Cons_mae,(2.95,maes.FP_Expert_Cons_mae+.05))
# ax.annotate(maes.GridIron_mae,(3.95,maes.GridIron_mae+.05))
# ax.annotate(maes.FantasyPros_mae,(4.95,maes.FantasyPros_mae+.05))
# plt.title('Week 8 MAE')
# fig.set_size_inches(12,8)
# fig.savefig(f'/Users/robertmegnia/Desktop/Week{week}_mae.png',dpi=100)