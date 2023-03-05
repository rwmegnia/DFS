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
game_date='2022-11-10'
season = 2022
week=10
method='Projection'
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
proj=proj[proj.full_name!='Jameis Winson']
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
#%%
# proj.loc[proj[method].isna()==True,method]=proj.loc[proj[method].isna()==True,'AvgPointsPerGame']
# proj=proj[proj[method]>4]
proj.drop(proj[(proj.position=='QB')&(proj.depth_team>1)].index,inplace=True)
proj.loc[proj['Roster Position']=='CPT',method]*=1.5
proj.loc[proj.Ceiling.isna()==True,'Ceiling']=proj.loc[proj.Ceiling.isna()==True,method]+2
proj.loc[proj.Floor.isna()==True,'Floor']=proj.loc[proj.Floor.isna()==True,method]+2
proj.loc[proj.RG_projection.isna()==True,'RG_projection']=proj.loc[proj.RG_projection.isna()==True,'wProjection']

proj.loc[proj['Roster Position']=='CPT','Ceiling']*=1.5
proj.loc[proj['Roster Position']=='CPT','Floor']*=1.5

proj.drop_duplicates(inplace=True)
proj=proj[proj[['salary','ID',method]].isna().any(axis=1)==False]
proj=proj[proj.full_name!='Matt Prater']

proj.drop(proj[(proj.position=='QB')&(proj['Roster Position']=='CPT')].index,inplace=True)
proj.min_deviation.fillna(0,inplace=True)
proj.max_deviation.fillna(proj.max_deviation.mean(),inplace=True)
# proj.drop(proj[(proj.position=='K')&(proj['Roster Position']=='CPT')].index,inplace=True)

players = proj.apply(
    lambda x: Player(
        player_id=x.ID,
        first_name=x.full_name.split(" ")[0],
        last_name=" ".join(x.full_name.split(" ")[1::]),
        positions=[x['Roster Position']],
        team=x.team,
        salary=x.salary,
        fppg=x[method],
        min_deviation=x.min_deviation,
        max_deviation=x.max_deviation,
        # fppg_ceil=x['Ceiling'],
        # fppg_floor=x['Floor'],
        ),
    axis=1,
        )
        # Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.FOOTBALL)
optimizer.player_pool.load_players(players.to_list())


player = optimizer.player_pool.get_player_by_name(
      "Cordarrelle Patterson", "CPT"
  )  # using player name and position
# player1= optimizer.player_pool.get_player_by_name(
#       "Lamar Jackson", "FLEX"
#   ) 
# # # # # # # player2= optimizer.player_pool.get_player_by_name(
# # # # # # #       "Deebo Samuel", "CPT"
# # # # # # #   ) 
players=[player]
for p in players:
    print(p)
    optimizer.player_pool.lock_player(p)
# # optimizer.player_pool.extend_players([player1,player2])
# # player = optimizer.player_pool.get_player_by_name(
# #       "Tyler Anderson","FLEX"
# #   )  # using player name and position
# optimizer.player_pool.remove_player(player)
# optimizer.set_deviation(100,10000)
# optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0.5,1.1))
# optimizer.set_min_salary_cap(50000)
# Execute Optimizer
LineupResults = []
n=0
for lineup in optimizer.optimize(n=10):
    print(lineup)
    n+=1
    lineup.actual_fantasy_points_per_game_projection=np.nan
    print(n)
# for lineup in optimizer.optimize(n=10):
#     print(lineup)
optimizer.print_statistic()
optimizer.export('/Users/robertmegnia/Desktop/ShowdownLineups.csv')
df=pd.read_csv('/Users/robertmegnia/Desktop/ShowdownLineups.csv')
df.sample(60).sample(20).to_csv('/Users/robertmegnia/Desktop/ShowdownLineups.csv',index=False)