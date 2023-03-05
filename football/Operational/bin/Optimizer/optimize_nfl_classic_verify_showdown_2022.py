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

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"

season = 2022
week = 7
game_date = "2022-10-20"
method = "wProjection"
# List Contst Slates
salary_dir = f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NFL/Showdown/{game_date}"

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
## Read in Projections

flex_projections = pd.read_csv(
    f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections_verified.csv"
)
flex_projections["Roster Position"] = "FLEX"
cpt_projections = pd.read_csv(
    f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections_verified.csv"
)
cpt_projections["Roster Position"] = "CPT"

proj = pd.concat([flex_projections, cpt_projections])
proj = filterProjections(proj, slate_salaries)
K_cpt=pd.read_csv(f'{datadir}/StartingLineupsRotoGrinders/2022_Week{week}_StartingLineups.csv')
K_cpt=K_cpt[K_cpt.position=='K']
K_cpt['Roster Position']='CPT'
K_cpt['game_time']=proj.game_time.unique()[0]
K_cpt['game_id']=proj.game_id.unique()[0]
K_cpt[method]=K_cpt.RG_projection
K_cpt=K_cpt[K_cpt.team.isin(proj.team)]
K_cpt.loc[K_cpt.team=='MIA','DKPts']=12
K_cpt.loc[K_cpt.team=='PIT','DKPts']=12

K_flex=pd.read_csv(f'{datadir}/StartingLineupsRotoGrinders/2022_Week{week}_StartingLineups.csv')
K_flex=K_flex[K_flex.position=='K']
K_flex['Roster Position']='FLEX'
K_flex['game_time']=proj.game_time.unique()[0]
K_flex['game_id']=proj.game_id.unique()[0]
K_flex[method]=K_flex.RG_projection
K_flex=K_flex[K_flex.team.isin(proj.team)]
K_flex.loc[K_flex.team=='MIA','DKPts']=12
K_flex.loc[K_flex.team=='PIT','DKPts']=12
proj=pd.concat([proj,K_cpt,K_flex])
# proj=proj[proj[method]>5]
proj.loc[proj["Roster Position"] == "CPT", method] *= 1.5
proj.loc[proj["Roster Position"] == "CPT", "DKPts"] *= 1.5

proj.drop(["salary", "ID"], axis=1, inplace=True)
proj = proj.merge(
    slate_salaries[["RotoName", "salary", "ID", "Roster Position"]],
    on=["RotoName", "Roster Position"],
    how="left",
)
#%%


# proj=pd.concat([proj,K])
proj.loc[proj.position == "DST", "full_name"] = proj.loc[
    proj.position == "DST", "full_name"
].apply(lambda x: x.upper() + " ")

# proj = proj[proj.FP_Proj.isna() == False]
proj["game_date"] = proj["game_time"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone(get_timezone())
    )
)

proj.fillna(0, inplace=True)
# Setup Optimizer
optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.FOOTBALL)

# Get Individual Game Information
game_info = {}
game_ids = proj[proj.game_id != 0].groupby("game_id", as_index=False).first()
for row in game_ids.iterrows():
    row = row[1]
    game_info[row.game_id] = GameInfo(
        row.home_team, row.away_team, row.game_date
    )

# Create Players List
# proj=proj[proj[method]>5]
# proj=proj[proj.position!='K']
players = proj.apply(
    lambda x: Player(
        player_id=x.ID,
        first_name=x.full_name.split(" ")[0],
        last_name=" ".join(x.full_name.split(" ")[1::]),
        positions=x["Roster Position"].split("/"),
        team=x.team,
        salary=x.salary,
        fppg=x[method],
        game_info=game_info[x.game_id],
        projected_ownership=x.AvgOwnership,
    ),
    axis=1,
)

optimizer.player_pool.load_players(players.to_list())
lock=optimizer.player_pool.get_player_by_name('Pat Freirmuth','CPT')
# lock=optimizer.player_pool.get_player_by_name('Brandon McManus','FLEX')
# lock=optimizer.player_pool.get_player_by_name('IND','FLEX')
# lock=optimizer.player_pool.get_player_by_name('DEN','FLEX')


optimizer.player_pool.lock_player(lock)
# optimizer.set_deviation(100, 100000)
# optimizer.set_min_salary_cap(50000)
AllLineupResults = []
LineupResults = []
for lineup in optimizer.optimize(
    n=150,randomness=True
):
    lineup_frame = print_lineups(lineup, proj)
    LineupResults.append(lineup.actual_fantasy_points_per_game_projection)
    AllLineupResults.append(lineup.actual_fantasy_points_per_game_projection)

print(f"Average Points: {np.mean(LineupResults)}")
print(f"Highest Score: {np.max(LineupResults)}")
print(np.mean(AllLineupResults))
print(np.max(AllLineupResults))
proj = proj[proj.RG_projection != 0]
