# -*- coding: utf-8 -*-
import pandas as pd
from pydfs_lineup_optimizer import Site, Sport, get_optimizer, PositionsStack, TeamStack
from pydfs_lineup_optimizer.fantasy_points_strategy import RandomFantasyPointsStrategy
import os
import warnings
from datetime import datetime
from ProcessRankings import processRankings
from optimizerTools import Locks, Exclusions, loadPlayers, mergeSwapProjectedPoints,getMaxLineupScore
import numpy as np

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
game_date = datetime.now().strftime("%Y-%m-%d")
# game_date='2022-07-15'
datadir = f"{basedir}/../../data"
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/MLB/Classic/{game_date}"
season = 2022
projdir = f"{basedir}/../../data/Projections/RealTime/{season}/Classic/{game_date}"
#%%
### Read in Projections
contestType='Classic'
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)

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
proj=proj.merge(slate_salaries[['RotoName','team','position','Roster Position','ID']],
                on=['RotoName','team','position','Roster Position'],
                how='left')
proj=proj[proj.ID.isna()==False]
proj = processRankings(proj, game_date, Scaled=False)
# Create optimizerti
if contestType == "Showdown":
    proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
    optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASEBALL)
else:
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASEBALL)


optimizer.load_players_from_csv("/Users/robertmegnia/Downloads/DKEntries.csv")
players = [
    mergeSwapProjectedPoints(p, proj,'ownership_proj')
    for p in optimizer.player_pool.all_players
    if mergeSwapProjectedPoints(p, proj,'ownership_proj') is not None
]
optimizer.player_pool.load_players(players)
# optimizer.player_pool.remove_player('Charlie Blackmon')
optimizer.restrict_positions_for_opposing_team(
    ["SP"], ["C", "1B", "2B", "3B", "SS", "OF"]
)
lineups = optimizer.load_lineups_from_csv("/Users/robertmegnia/Downloads/DKEntries.csv")
optimizer.restrict_positions_for_opposing_team(
    ["SP","P","RP"], [c for c in proj['Roster Position'].unique() if c not in['RP','SP','P']]
)
# optimizer.player_pool.exclude_teams(['DET','CLE','STL','CIN'])
Exclude = ['Jacob Degrom']
# Exclusions(Exclude, optimizer)

Include = ["Carolos Rodon"]
# Locks(Include, optimizer)
for lineup in optimizer.optimize_lineups(lineups):
    lineup.actual_fantasy_points_per_game_projection='None'
    print(lineup)
slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
optimizer.export(f"{datadir}/ExportedLineups/{slate}_LineupsLateSwap.csv")
lineups = pd.read_csv(f"{datadir}/ExportedLineups/{slate}_LineupsLateSwap.csv")
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
