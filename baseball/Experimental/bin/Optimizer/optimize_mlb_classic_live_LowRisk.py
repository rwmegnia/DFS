# -*- coding: utf-8 -*-
import pandas as pd
from pydfs_lineup_optimizer import (
    Site,
    Sport,
    get_optimizer,
    PositionsStack,
    TeamStack,
)
from pydfs_lineup_optimizer.fantasy_points_strategy import (
    RandomFantasyPointsStrategy,
)
from pydfs_lineup_optimizer.exposure_strategy import AfterEachExposureStrategy

import os
import warnings
from datetime import datetime
from optimizerTools import *
import numpy as np
from ProcessRankings import processRankings
from ProcessSlateStats import processSlateStats
warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
game_date = datetime.now().strftime("%Y-%m-%d")
# game_date='2022-09-05'
datadir = f"{basedir}/../../data"
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/MLB/Classic/{game_date}"
season = 2022
projdir = f"{basedir}/../../data/Projections/RealTime/{season}/Classic/{game_date}"
#%%
### Read in Projections
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)
# proj.loc[proj.ownership_proj==0,'ownership_proj']=proj.loc[proj.ownership_proj==0,'RG_projection']

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
proj.drop('ID',axis=1,inplace=True)
proj=proj.merge(slate_salaries[['RotoName','team','position','Roster Position','ID','game_time']],
                on=['RotoName','team','position','Roster Position','game_time'],
                how='left')
proj=proj[proj.ID.isna()==False]
# proj.loc[proj.full_name=='Bobby Wittjr','Salary']=4700
# proj = processRankings(proj, game_date, Scaled=True)
processed = processSlateStats(proj,projdir,slate)
#%%
# proj=proj[proj.order<=]
# proj.loc[proj.full_name=='Oneil Cruz','position']='SS'
# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASEBALL)
# proj=processRankings(proj, game_date,Scaled=True)
proj["game_date"] = pd.to_datetime(proj.game_date)
# proj["Projection"] = proj[
#     ["Stochastic", "ML", "RG_projection", "PMM"]
# ].mean(axis=1)
proj.loc[proj.Ceiling.isna() == True, "Ceiling"] = proj.loc[
    proj.Ceiling.isna() == True, "Projection"
]
proj.loc[proj.Stochastic.isna() == True, "Stochastic"] = proj.loc[
    proj.Stochastic.isna() == True, "Projection"
]
players = loadPlayers(proj, "ownership_proj", "Classic")
optimizer.player_pool.load_players(players.to_list())
#%%
# optimizer.set_max_repeating_players(3)
optimizer.set_min_salary_cap(49500)
# optimizer.add_stack(
#     TeamStack(
#         5, for_teams=['MIN','CLE','NYY','LAD','CWS'],for_positions=["C", "1B", "2B", "3B", "SS", "OF"]
#     )
# )
# optimizer.add_stack(
#     TeamStack(
#         5, for_teams=["ATL"], for_positions=["C", "1B", "2B", "3B", "SS", "OF"]
#     )
# )

optimizer.restrict_positions_for_opposing_team(
    ["SP","P","RP"], [c for c in proj['Roster Position'].unique() if c not in['RP','SP','P']]
)
optimizer.player_pool.exclude_teams(['PHI','WAS'])
Exclude = ['Omar Navarez',"Jacob Degrom"]
# Exclusions(Exclude, optimizer)

Include = ["Aaron Judge"

           ]
# Locks(Include, optimizer)
# optimizer.player_pool.lock_player("Julio Rodriguez")
# optimizer.player_pool.exclude_teams()
# Execute Optimizer
# optimizer.set_deviation(0.5, 1)
for lineup in optimizer.optimize(10):
    lineup.actual_fantasy_points_per_game_projection=None
    print(lineup)
slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_LowRiskLineups.csv")
lineups = pd.read_csv(f"{datadir}/ExportedLineups/{slate}_LowRiskLineups.csv")
batter_db = pd.read_csv(f"{datadir}/game_logs/batterstatsDatabase.csv")
pitcher_db = pd.read_csv(f"{datadir}/game_logs/pitcherStatsDatabase.csv")
lineups["maxScore"] = lineups.T.apply(
    lambda x: getMaxLineupScore(x, proj, pitcher_db, batter_db)
)
lineups.sort_values(by="maxScore", ascending=False, inplace=True)
lineups.to_csv(
    f"{datadir}/ExportedLineups/{slate}_LowRiskLineups.csv", index=False
)
print(lineups[lineups.maxScore == lineups.maxScore.max()].T)
