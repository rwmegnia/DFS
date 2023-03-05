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
from exportNicksAggs import exportNicksAggs2DB
from exportMillWinner import exportMilliWinner
from OptimizerTools import *
import os
import warnings
from datetime import datetime

game_date = "2023-01-07"
season = 2022
week = 18
method='NicksAgg'
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
proj=proj.groupby('gsis_id',as_index=False).last()
proj['Ceiling']=proj[['Ceiling','FP_Proj_Ceil']].mean(axis=1)

if week==8:
    proj.loc[proj.full_name=='Sam Ehlinger','depth_team']=1
proj.loc[proj.full_name=='Andy Dalton','depth_team']=1
proj.loc[proj.full_name=='Jameis Winston','depth_team']=2
proj.drop(proj[(proj.position=='QB')&(proj.depth_team!=1)].index,inplace=True)
proj.loc[proj.Floor.isna()==True,'Floor']=proj.loc[proj.Floor.isna()==True,method]/1.5
proj.loc[proj.Ceiling.isna()==True,'Ceiling']=proj.loc[proj.Ceiling.isna()==True,method]*1.5
proj.loc[proj[method]<proj.Floor,'Floor']=proj.loc[proj[method]<proj.Floor,method]-(proj.loc[proj[method]<proj.Floor,'Floor']/2)
proj.loc[proj[method]>proj.Ceiling,'Ceiling']=proj.loc[proj[method]>proj.Ceiling,method]+(proj.loc[proj[method]>proj.Ceiling,'Ceiling']/2)
proj.loc[proj[method].isna()==True,method]=proj.loc[proj[method].isna()==True,'wProjection']
proj['Ceiling']=proj[[f'{method}','Ceiling']].mean(axis=1)
proj['Floor']=proj[[f'{method}','Floor']].mean(axis=1)


slate_salaries = getSlateSalaries(salary_dir, game_date)
proj = filterProjections(proj, slate_salaries)
#%%
# Get Individual Game Information
game_info = {}
game_ids = proj.groupby("game_id", as_index=False).first()
for row in game_ids.iterrows():
    row = row[1]
    game_info[row.game_id] = GameInfo(
        row.home_team, row.away_team, row.game_date
    )
    # Create list of pydfs-lineup-optimizer Player objects


players = proj.apply(
    lambda x: Player(
        player_id=int(x.ID),
        first_name=x.full_name.split(" ")[0],
        last_name=" ".join(x.full_name.split(" ")[1::]),
        positions=[x.position],
        team=x.team,
        salary=x.salary,
        fppg=x[method],
        fppg_floor=x['Floor'],
        fppg_ceil=x['Ceiling'],
        # min_deviation=x.min_deviation,
        # max_deviation=x.max_deviation,
        projected_ownership=x.AvgOwnership / 100,
        is_injured=False,
        game_info=game_info[x.game_id],
    ),
    axis=1,
)

# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
optimizer.player_pool.load_players(players.to_list())

optimizer.restrict_positions_for_opposing_team(
    ["DST"], ["QB", "RB", "WR", "TE"]
)
optimizer.set_max_repeating_players(7)
optimizer.set_min_salary_cap(49500)
optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0.25,0.75))
Exclude=['Sam Ehlinger']
# Exclusions(Exclude,optimizer)
LineupResults = []
N_lineups=500
lineup_iteration=1
total_lineups=0
lineups=set([])
rb_flex=True
while total_lineups<10000:
    if rb_flex==True:
        optimizer.set_players_with_same_position({'RB':1})
    else:
        optimizer.set_players_with_same_position({'WR':1})
    for lineup in optimizer.optimize(n=N_lineups,randomness=True):
        lineup.actual_fantasy_points_per_game_projection = None
        print(lineup)
        print(lineup_iteration)
        lineup_iteration+=1
    lineups.update(optimizer.last_context.lineups)
    total_lineups+=500
    if rb_flex==True:
        rb_flex=False
    else:
        rb_flex=True
optimizer.last_context.lineups=lineups
optimizer.print_statistic()
optimizer.export('NicksLineups.csv')
exportNicksAggs2DB(len(optimizer.last_context.lineups),season,week,proj)


