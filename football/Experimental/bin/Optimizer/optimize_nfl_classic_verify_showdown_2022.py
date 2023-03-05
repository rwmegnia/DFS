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
from pydfs_lineup_optimizer.stacks import GameStack,PlayersGroup
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
week = 17
game_date = "2022-12-29"
method = "DKPts"
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
# pool=pd.read_csv(
#     f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_MNF_Showdown.csv"
# )
# proj=proj[proj.gsis_id.isin(pool.gsis_id)]
proj = filterProjections(proj, slate_salaries)
K_cpt=pd.read_csv(f'{datadir}/StartingLineupsRotoGrinders/2022_Week{week}_StartingLineups.csv')
K_cpt=K_cpt[K_cpt.position=='K']
K_cpt['Roster Position']='CPT'
K_cpt['game_time']=proj.game_time.unique()[0]
K_cpt['game_id']=proj.game_id.unique()[0]
K_cpt[method]=K_cpt.RG_projection
K_cpt=K_cpt[K_cpt.team.isin(proj.team)]
K_cpt.loc[K_cpt.team=='DAL','DKPts']=10
K_cpt.loc[K_cpt.team=='TEN','DKPts']=7

K_flex=pd.read_csv(f'{datadir}/StartingLineupsRotoGrinders/2022_Week{week}_StartingLineups.csv')
K_flex=K_flex[K_flex.position=='K']
K_flex['Roster Position']='FLEX'
K_flex['game_time']=proj.game_time.unique()[0]
K_flex['game_id']=proj.game_id.unique()[0]
K_flex[method]=K_flex.RG_projection
K_flex=K_flex[K_flex.team.isin(proj.team)]
K_flex.loc[K_flex.team=='DAL','DKPts']=10
K_flex.loc[K_flex.team=='TEN','DKPts']=7
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
# position!='K']

proj.min_deviation.fillna(0,inplace=True)
proj.max_deviation.fillna(proj.max_deviation.mean(),inplace=True)
proj.drop(proj[(proj.position=='K')&(proj['Roster Position']=='CPT')].index,inplace=True)
proj['Ceiling']=proj[[f'{method}','Ceiling']].mean(axis=1)
proj['Floor']=proj[[f'{method}','Floor']].mean(axis=1)
proj['ID']=proj.ID.astype(int)
pool = pd.read_csv('/Users/robertmegnia/Desktop/TNF_Showdown.csv')
proj=proj[proj.ID.isin(pool.ID)]
players = proj.apply(
    lambda x: Player(
        player_id=x.ID,
        first_name=x.full_name.split(" ")[0],
        last_name=" ".join(x.full_name.split(" ")[1::]),
        positions=x["Roster Position"].split("/"),
        team=x.team,
        salary=x.salary,
        fppg=x[method],
        min_deviation=x['min_deviation'],
        max_deviation=x['max_deviation'],
        fppg_floor=x['Floor'],
        fppg_ceil=x['Ceiling'],
        game_info=game_info[x.game_id],
        projected_ownership=x.AvgOwnership,
    ),
    axis=1,
)

optimizer.player_pool.load_players(players.to_list())
team1=proj.team.unique()[0]
qb1=proj[(proj.team==team1)&(proj.position=='QB')].full_name.values[0]
qb1=optimizer.player_pool.get_player_by_name(qb1,'FLEX')
# dst1=proj[(proj.team==team1)&(proj.position=='DST')].full_name.values[0]
# dst1=optimizer.player_pool.get_player_by_name(dst1,'CPT')
team2=proj.team.unique()[1]
qb2=proj[(proj.team==team2)&(proj.position=='QB')].full_name.values[0]
qb2=optimizer.player_pool.get_player_by_name(qb2,'FLEX')
# dst2=proj[(proj.team==team2)&(proj.position=='DST')].full_name.values[0]
# dst2=optimizer.player_pool.get_player_by_name(dst2,'CPT')
# team1 Receiviers
team1_receivers=proj[(proj.team==team1)&(proj['Roster Position']=='CPT')&(proj.position.isin(['WR','TE']))]
team1_receivers=[c for c in team1_receivers.apply(lambda x: optimizer.player_pool.get_player_by_name(x.full_name,x['Roster Position']),axis=1)]

# team1 RBs
team1_rbs=proj[(proj.team==team1)&(proj['Roster Position']=='FLEX')&(proj.position=='RB')]
team1_rbs=[c for c in team1_rbs.apply(lambda x: optimizer.player_pool.get_player_by_name(x.full_name,x['Roster Position']),axis=1)]

# team2 Receivers
team2_receivers=proj[(proj.team==team2)&(proj['Roster Position']=='CPT')&(proj.position.isin(['WR','TE']))]
team2_receivers=[c for c in team2_receivers.apply(lambda x: optimizer.player_pool.get_player_by_name(x.full_name,x['Roster Position']),axis=1)]

# team2 RBs
team2_rbs=proj[(proj.team==team2)&(proj['Roster Position']=='FLEX')&(proj.position=='RB')]
team2_rbs=[c for c in team2_rbs.apply(lambda x: optimizer.player_pool.get_player_by_name(x.full_name,x['Roster Position']),axis=1)]


for receiver in team1_receivers:
    group = PlayersGroup(
        [qb1],
        max_from_group=1,
        depends_on=receiver,
        strict_depend=False, 
    )
    optimizer.add_players_group(group)

for receiver in team2_receivers:
    group = PlayersGroup(
        [qb2],
        max_from_group=1,
        depends_on=receiver,
        strict_depend=False, 
    )
    optimizer.add_players_group(group)


# lock=optimizer.player_pool.get_player_by_name('Dalton Shultz','CPT')
# optimizer.player_pool.lock_player(lock)
# optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0.25,0.85))
# optimizer.set_min_salary_cap(49000)
AllLineupResults = []
LineupResults = []
for lineup in optimizer.optimize(n=1):
    lineup_frame = print_lineups(lineup, proj)
    LineupResults.append(lineup.actual_fantasy_points_per_game_projection)
    AllLineupResults.append(lineup.actual_fantasy_points_per_game_projection)

print(f"Average Points: {np.mean(LineupResults)}")
print(f"Highest Score: {np.max(LineupResults)}")
print(np.mean(AllLineupResults))
print(np.max(AllLineupResults))
proj = proj[proj.RG_projection != 0]
