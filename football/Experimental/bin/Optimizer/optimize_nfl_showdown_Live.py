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
from pydfs_lineup_optimizer.stacks import GameStack,PositionsStack,PlayersGroup
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
game_date='2023-02-12'
season = 2022
week=22
method='wProjection'
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NFL/Showdown/{game_date}"
projdir = f"{basedir}/../../data/Projections/RealTime/{season}/Classic/{game_date}"
warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
# from getConsensusRank import getConsensusRanking

# Read in Projections data frame and create player pool
flex_projections = pd.read_csv(
            f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections.csv"
        )
flex_projections['Roster Position']='FLEX'
cpt_projections=pd.read_csv(
            f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections.csv"
        )
cpt_projections['Roster Position']='CPT'
proj=pd.concat([flex_projections,cpt_projections])
proj.loc[proj[method].isna()==True,method]=proj.loc[proj[method].isna()==True,'wProjection']
# proj=proj[proj.full_name!='Matt Prater']
proj.drop(['salary','ID'],axis=1,inplace=True)
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
slate_salaries.loc[slate_salaries.position == "DST", "full_name"] = slate_salaries.loc[
    slate_salaries.position == "DST", "team"
]
slate_salaries.loc[slate_salaries.position == "DST", "RotoName"] = slate_salaries.loc[
    slate_salaries.position == "DST", "team"
]
proj=proj[proj.team.isin(slate_salaries.team)]
K_cpt=pd.read_csv(f'{datadir}/StartingLineupsRotoGrinders/2022_Week{week}_StartingLineups.csv')
K_cpt=K_cpt[K_cpt.position=='K']
K_cpt['Roster Position']='CPT'
K_cpt['game_time']=proj.game_time.unique()[0]
# K_cpt['game_id']=proj.game_id.unique()[0]
K_cpt[method]=K_cpt.RG_projection
K_cpt=K_cpt[K_cpt.team.isin(proj.team)]

K_flex=pd.read_csv(f'{datadir}/StartingLineupsRotoGrinders/2022_Week{week}_StartingLineups.csv')
K_flex=K_flex[K_flex.position=='K']
K_flex['Roster Position']='FLEX'
K_flex['game_time']=proj.game_time.unique()[0]
# K_flex['game_id']=proj.game_id.unique()[0]
K_flex[method]=K_flex.RG_projection
K_flex=K_flex[K_flex.team.isin(proj.team)]
proj=pd.concat([proj,K_cpt,K_flex])
proj=proj.merge(slate_salaries[['position','ID','Roster Position','salary','team','RotoName','AvgPointsPerGame']],on=['position','Roster Position','team','RotoName'],how='left')
# proj.drop(proj[(proj.position=='QB')&(proj.depth_team>1)].index,inplace=True)
proj.loc[proj['Roster Position']=='CPT',method]*=1.5
proj.loc[proj.Ceiling.isna()==True,'Ceiling']=proj.loc[proj.Ceiling.isna()==True,method]*1.5
proj.loc[proj.Floor.isna()==True,'Floor']=proj.loc[proj.Floor.isna()==True,method]*.5
proj.loc[proj.RG_projection.isna()==True,'RG_projection']=proj.loc[proj.RG_projection.isna()==True,'wProjection']

proj.loc[proj['Roster Position']=='CPT','Ceiling']*=1.5
proj.loc[proj['Roster Position']=='CPT','Floor']*=1.5

# proj.drop_duplicates(inplace=True)
# proj=proj[proj[['salary','ID',method]].isna().any(axis=1)==False]

# proj['ID']=proj.ID.astype(int)
proj=pd.read_csv(f'/Users/robertmegnia/Desktop/SuberBowl.csv')
proj['min_exposure']=0
proj['max_exposure']=1
# proj.loc[(proj.full_name=='Jamal Agnew')&(proj['Roster Position']=='FLEX'),'min_exposure']=0.30
# proj=proj[proj.full_name!='James Robinson']
#%%
players = proj.apply(
    lambda x: Player(
        player_id=x.ID,
        first_name=x.full_name.split(" ")[0],
        last_name=" ".join(x.full_name.split(" ")[1::]),
        positions=[x['Roster Position']],
        team=x.team,
        min_exposure=x.min_exposure,
        max_exposure=x.max_exposure,
        salary=x.salary,
        fppg=x[method],
        ),
    axis=1,
        )
        # Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.FOOTBALL)
optimizer.player_pool.load_players(players.to_list())
team1=proj.team.unique()[0]
qb1=proj[(proj.team==team1)&(proj.position=='QB')].full_name.values[0]
qb1=optimizer.player_pool.get_player_by_name(qb1,'FLEX')
dst1=proj[(proj.team==team1)&(proj.position=='DST')].full_name.values[0]
dst1=optimizer.player_pool.get_player_by_name(dst1,'FLEX')
team2=proj.team.unique()[1]
qb2=proj[(proj.team==team2)&(proj.position=='QB')].full_name.values[0]
qb2=optimizer.player_pool.get_player_by_name(qb2,'FLEX')
dst2=proj[(proj.team==team2)&(proj.position=='DST')].full_name.values[0]
dst2=optimizer.player_pool.get_player_by_name(dst2,'FLEX')


# team1 Receiviers
team1_receivers=proj[(proj.team==team1)&(proj['Roster Position']=='CPT')&(proj.position.isin(['WR','TE']))]
team1_receivers=[c for c in team1_receivers.apply(lambda x: optimizer.player_pool.get_player_by_name(x.full_name,x['Roster Position']),axis=1)]
# team1 RBs
team1_rbs=proj[(proj.team==team1)&(proj['Roster Position']=='CPT')&(proj.position=='RB')]
team1_rbs=[c for c in team1_rbs.apply(lambda x: optimizer.player_pool.get_player_by_name(x.full_name,x['Roster Position']),axis=1)]


# team2 Receivers
team2_receivers=proj[(proj.team==team2)&(proj['Roster Position']=='CPT')&(proj.position.isin(['WR','TE']))]
team2_receivers=[c for c in team2_receivers.apply(lambda x: optimizer.player_pool.get_player_by_name(x.full_name,x['Roster Position']),axis=1)]
# team2 RBs
team2_rbs=proj[(proj.team==team2)&(proj['Roster Position']=='CPT')&(proj.position=='RB')]
team2_rbs=[c for c in team2_rbs.apply(lambda x: optimizer.player_pool.get_player_by_name(x.full_name,x['Roster Position']),axis=1)]

for rb in team1_rbs:
        group = PlayersGroup(
            [dst1,qb1],
            min_from_group=2,
            depends_on=rb,
            strict_depend=False, 
        )
        optimizer.add_players_group(group)
for rb in team2_rbs:
        group = PlayersGroup(
            [dst2,qb1],
            min_from_group=2,
            depends_on=rb,
            strict_depend=False, 
        )
        optimizer.add_players_group(group)
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
    

# player = optimizer.player_pool.get_player_by_name(
#       "Joe Burrow", "FLEX"
#   )  # using player name and position
# player2 = optimizer.player_pool.get_player_by_name(
#       "Tyler Boyd", "CPT"
#   )  
# player3 = optimizer.player_pool.get_player_by_name(
#         "Patrick Mahomes", "FLEX"
#     )  
# players=[player,player2,player3]
# for p in players:
#     print(p)
#     optimizer.player_pool.lock_player(p)
# optimizer.player_pool.extend_players([player1,player2])
# player = optimizer.player_pool.get_player_by_name(
#       "Tyler Anderson","FLEX"
#   )  # using player name and position
# optimizer.player_pool.remove_player(player)
# optimizer.set_deviation(100,10000)
optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0.25,0.85))
# # optimizer.set_min_salary_cap(50000)
# #Execute Optimizer
# LineupResults = []
# n=0
for lineup in optimizer.optimize(n=1000):
    print(lineup)
    n+=1
    lineup.actual_fantasy_points_per_game_projection=np.nan
    print(n)
optimizer.print_statistic()
optimizer.export('ShowdownLineups.csv')
df=pd.read_csv('ShowdownLineups.csv')
df.sample(191).to_csv('/Users/robertmegnia/Desktop/ShowdownLineups.csv',index=False)