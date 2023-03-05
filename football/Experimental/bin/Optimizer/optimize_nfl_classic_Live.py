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
from pydfs_lineup_optimizer.stacks import GameStack, Stack
from pydfs_lineup_optimizer.fantasy_points_strategy import *
from pydfs_lineup_optimizer.solvers.pulp_solver import PuLPSolver
from pydfs_lineup_optimizer.fantasy_points_strategy import (
    ProgressiveFantasyPointsStrategy,
)
from pydfs_lineup_optimizer import Player, PlayersGroup
from pydfs_lineup_optimizer.player import GameInfo, Player
from pydfs_lineup_optimizer.tz import get_timezone
from pytz import timezone
from OptimizerTools import *
import os
import warnings
from datetime import datetime


game_date = "2023-01-29"
season = 2022
week = 21
method='wProjection'
salary_dir = (
    f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NFL/Classic/{game_date}"
)
projdir = (
    f"{basedir}/../../data/Projections/RealTime/{season}/Classic/{game_date}"
)
warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
# from getConsensusRank import getConsensusRanking

# Read in Projections data frame and create player pool
proj = pd.read_csv(
    f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections.csv"
)
# pool=pd.read_csv(
#     f"{datadir}/PlayerPools/Week{week}_PlayerPool_Saturday.csv"
# # )
pool=pd.read_csv(
    f"{datadir}/PlayerPools/Week{week}_PlayerPool.csv"
)
dst=proj[proj.position=='DST']
proj=proj[proj.position!='DST']
proj.drop(['wProjection','Floor','Ceiling'],axis=1,inplace=True)
proj=proj.merge(pool[['gsis_id','wProjection','Floor','Ceiling']],on='gsis_id',how='left')
proj=pd.concat([proj[proj.gsis_id.isin(pool.gsis_id)],dst])
#%%
proj.drop(proj[(proj.position=='QB')&(proj.depth_team!=1)].index,inplace=True)
slate_salaries=getSlateSalaries(salary_dir, game_date)
proj.loc[proj[method].isna()==True,method]=proj.loc[proj[method].isna()==True,'wProjection']
proj = filterProjections(proj, slate_salaries)
# Get Individual Game Information
game_info = {}
game_ids = proj.groupby("game_id", as_index=False).first()
for row in game_ids.iterrows():
    row = row[1]
    game_info[row.game_id] = GameInfo(
        row.home_team, row.away_team, row.game_date
    )
    # Create list of pydfs-lineup-optimizer Player objects
proj["max_exposure"] = 1.0
proj["min_exposure"] = 0.0
proj['Ceiling2']=(proj.wProjection+(proj.wProjection*proj.max_deviation))
proj['Ceiling']=proj[['Ceiling2','Ceiling']].mean(axis=1)
proj['Floor']=proj[['wProjection','Floor']].mean(axis=1)

players = proj.apply(
    lambda x: Player(
        player_id=int(x.ID),
        first_name=x.full_name.split(" ")[0],
        last_name=" ".join(x.full_name.split(" ")[1::]),
        positions=[x.position],
        team=x.team,
        salary=x.salary,
        fppg=x[method],
        fppg_floor=x.Floor,
        fppg_ceil=x.Ceiling,
        projected_ownership=x.AvgOwnership / 100,
        is_injured=False,
        game_info=game_info[x.game_id],
        max_exposure=x.max_exposure,
        min_exposure=x.min_exposure,
    ),
    axis=1,
)

# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
optimizer.player_pool.load_players(players.to_list())
pass_teams=getPassingStackTeams(proj)
rb_teams=getRunningBackTeams(proj)
Include = ['Travis Kelce','Joe Burrow']
Locks(Include,optimizer)
# optimizer.add_stack(GameStack(4))
# optimizer.set_players_with_same_position({"WR": 1})
# optimizer.set_max_repeating_players(6)
# optimizer.restrict_positions_for_opposing_team(
#     ["DST"], ["QB","RB","WR","TE"]
# )
# optimizer.set_min_salary_cap(49200)


#JAX Group
# group = PlayersGroup(
#     optimizer.player_pool.get_players('Zay Jones',
#                                       'Christian Kirk',
#                                       'Evan Engram',
#                                       'Marvin Jones'),
#     min_from_group=2,
#     max_from_group=3,
#     depends_on=optimizer.player_pool.get_player_by_name('Trevor Lawrence'),
#     strict_depend=False,
# )
# optimizer.add_players_group(group)

#JAX Runback Group
# group = PlayersGroup(
#     optimizer.player_pool.get_players('Travis Kelce',
#                                       'Juju Smitchschuster',
#                                       'Justin Watson',
#                                       'Kadarius Toney',
#                                       'Marquez Valdesscantling',
#                                       'Jerick Mckinnon'),
#     min_from_group=2,
#     max_from_group=2,
#     depends_on=optimizer.player_pool.get_player_by_name('Trevor Lawrence'),
#     strict_depend=False,
# )
# optimizer.add_players_group(group)
# 
# optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0,0.25))
# Execute Optimizer
LineupResults = []
N_lineups=10
lineup_iteration=1
total_lineups=0
lineups=set([])
rb_flex=True
while total_lineups<10:
    for lineup in optimizer.optimize(n=N_lineups):
        lineup.actual_fantasy_points_per_game_projection = None
        print(lineup)
        print(lineup_iteration)
        lineup_iteration+=1
    lineups.update(optimizer.last_context.lineups)
    total_lineups+=100
    if rb_flex==True:
        rb_flex=False
    else:
        rb_flex=True
optimizer.last_context.lineups=lineups
optimizer.print_statistic()
optimizer.export('Lineups.csv')
df=pd.read_csv('Lineups.csv')
df.drop_duplicates(inplace=True)
df.sample(20).to_csv('/Users/robertmegnia/Desktop/NFL_Lineups.csv',index=False)

