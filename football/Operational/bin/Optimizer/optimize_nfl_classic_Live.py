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


game_date = "2022-11-10"
season = 2022
week = 10
method='Projection'
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
if week==8:
    proj.loc[proj.full_name=='Sam Ehlinger','depth_team']=1
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


#%%
proj=proj[proj[method]>5]
players = proj.apply(
    lambda x: Player(
        player_id=int(x.ID),
        first_name=x.full_name.split(" ")[0],
        last_name=" ".join(x.full_name.split(" ")[1::]),
        positions=[x.position],
        team=x.team,
        salary=x.salary,
        fppg=x[method],
        projected_ownership=x.AvgOwnership / 100,
        is_injured=False,
        game_info=game_info[x.game_id],
        max_exposure=x.max_exposure,
        min_exposure=x.min_exposure,
        # min_deviation=x.min_deviation,
        # max_deviation=x.max_deviation
    ),
    axis=1,
)

pass_teams = getPassingStackTeams(proj)
rb_teams = getRunningBackTeams(proj)

# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
optimizer.player_pool.load_players(players.to_list())
Exclude = ["Noah Fant"]
# Exclusions(Exclude,optimizer)

Include = [
    "Patrick Mahomes",
    "Travis Kelce"
]
# Locks(Include, optimizer)
# optimizer.add_stack(PositionsStack(["QB","WR",'TE']))
optimizer.set_players_with_same_position({"RB": 1})
# optimizer.add_stack(PositionsStack(["RB"],for_teams=rb_teams))
optimizer.restrict_positions_for_opposing_team(
    ["DST"], ["QB", "RB", "WR", "TE"]
)
# optimizer.set_players_from_one_team({'JAX':1})
# optimizer.player_pool.exclude_teams(['NO'])
# optimizer.restrict_positions_for_same_team(("RB","TE"))
# optimizer.add_players_group(group)
# optimizer.add_stack(Stack([mahomes_kelce_group]))
# optimizer.set_players_from_one_team(2)
# optimizer.add_stack(GameStack(4))
# optimizer.player_pool.exclude_teams(['JAX','CHI','ATL','SEA'])
# e=optimizer.set_projected_ownership(12,17)
# optimizer.force_positions_for_opposing_team(("QB", "WR"))
optimizer.set_min_salary_cap(49800)
optimizer.set_max_repeating_players(7)
# optimizer.restrict_positions_for_same_team(('WR','RB'),("WR","TE"),("RB","TE"))
# optimizer.set_deviation(0, 0.02)
optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0,0.25))
# Execute Optimizer
LineupResults = []
N=1
lineups=set({})
for lineup in optimizer.optimize(n=150):
    lineup.actual_fantasy_points_per_game_projection = None
    print(lineup)
    N+=1
# lineups.update(optimizer.last_context.lineups)
# optimizer.set_players_with_same_position({'WR':1})
# for lineup in optimizer.optimize(n=75,exclude_lineups=lineups,randomness=True):
#     lineup.actual_fantasy_points_per_game_projection = None
#     print(lineup)
#     N+=1
# lineups.update(optimizer.last_context.lineups)
# optimizer.last_context.lineups=lineups
optimizer.print_statistic()
optimizer.export('Lineups.csv')
# df=pd.read_csv('Lineups.csv')
# df=df.sample(20)
# df.to_csv('/Users/robertmegnia/Desktop/Lineups.csv',index=False)
