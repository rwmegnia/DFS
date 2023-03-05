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
from pydfs_lineup_optimizer.stacks import GameStack,Stack
from pydfs_lineup_optimizer.fantasy_points_strategy import *
from pydfs_lineup_optimizer.solvers.pulp_solver import PuLPSolver
from pydfs_lineup_optimizer.fantasy_points_strategy import (
    ProgressiveFantasyPointsStrategy,
)
from pydfs_lineup_optimizer import Player,PlayersGroup
from pydfs_lineup_optimizer.player import GameInfo, Player
from pydfs_lineup_optimizer.tz import get_timezone
from pytz import timezone
from OptimizerTools import *
import os
import warnings
from datetime import datetime
game_date='2022-12-15'
season = 2022
week=15
method='wProjection'
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NFL/Classic/{game_date}"
projdir = f"{basedir}/../../data/Projections/RealTime/{season}/Classic/{game_date}"
warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
# from getConsensusRank import getConsensusRanking

# Read in Projections data frame and create player pool
proj = pd.read_csv(
            f"{datadir}/Projections/{season}/WeeklyProjections/{season}_Week{week}_Projections.csv"
        )
slates = [
    file
    for file in os.listdir(
        f"{salary_dir}"
    )
    if (".csv" in file)&(f'{game_date}_salaries.csv' not in file)
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
slate_salaries.loc[slate_salaries.position=='DST','RotoName']=slate_salaries.loc[slate_salaries.position=='DST','team']
proj=filterProjections(proj, slate_salaries)
proj['ID']=proj.ID.astype(int)
# Get Individual Game Information
game_info = {}
game_ids = proj.groupby("game_id", as_index=False).first()
for row in game_ids.iterrows():
    row = row[1]
    game_info[row.game_id] = GameInfo(
        row.home_team, row.away_team, row.game_date
        )
        # Create list of pydfs-lineup-optimizer Player objects
proj['max_exposure']=1.
proj['min_exposure']=0.
# proj.loc[proj.full_name=='Michael Pittman','max_exposure']=0.5

# pass_teams = getPassingStackTeams(proj)
# # pass_teams=['KC','LV','LAC','CIN']
# rb_teams = getRunningBackTeams(proj)

# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.FOOTBALL)
optimizer.load_players_from_csv("/Users/robertmegnia/Downloads/DKEntries.csv")
# proj.loc[proj.full_name=='Joe Burrow',method]=20.08
# proj.loc[proj.full_name=='Tee Higgins',method]=28.40
# proj.loc[proj.full_name=='Jamarr Chase',method]=12.10
players = [
    mergeSwapProjectedPoints(p, proj,method)
    for p in optimizer.player_pool.all_players
    if mergeSwapProjectedPoints(p, proj,'ownership_proj') is not None
]
optimizer.player_pool.load_players(players)
lineups=optimizer.load_lineups_from_csv("/Users/robertmegnia/Downloads/DKEntries.csv")
# Exclude=['Rondale Moore','KJ Hamler','Kadarius Toney','Jakobi Meyers','Richie Jamesjr']
# Exclusions(Exclude,optimizer)
# group = PlayersGroup(
#     optimizer.player_pool.get_players('Tj Hockenson','Josh Reynolds','Jamaal Williams','Kalif Raymond'),
#     max_from_group=2,
# )
# optimizer.add_players_group(group)
# mahomes_kelce_group = PlayersGroup(optimizer.player_pool.get_players('Patrick Mahomes','Travis Kelce','Joe Mixon'),max_exposure=0.5)
# optimizer.add_stack(Stack([mahomes_kelce_group]))
# Include=['Patrick Mahomes',
#          'Travis Kelce',
#          'Joe Mixon',
#          'cin ']
# Locks(Include,optimizer)

# optimizer.add_stack(
#     PositionsStack(["QB", "WR"])
#     )
# optimizer.player_pool.exclude_teams(['CHI','NYG','MIA'])
optimizer.set_players_with_same_position({"RB": 1})
# optimizer.add_stack(PositionsStack(["RB"],for_teams=rb_teams))
# # optimizer.add_stack(PositionsStack(["RB"],for_teams=teams))
optimizer.restrict_positions_for_opposing_team(
            ["DST"], ["QB", "RB", "WR", "TE"]
        )
# optimizer.set_projected_ownership(11,14)

# optimizer.player_pool.lock_player('Joe Burrow')
# optimizer.player_pool.lock_player('Dalvin Cook')
# optimizer.force_positions_for_opposing_team(("QB", "WR"))
# optimizer.set_min_salary_cap(49700)
# Execute Optimizer
LineupResults = []
# Locks(Include, optimizer)
for lineup in optimizer.optimize_lineups(lineups):
    lineup.actual_fantasy_points_per_game_projection=None
    print(lineup)
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_LineupsLateSwap.csv")

# optimizer.export('Week1_Lineups.csv')
