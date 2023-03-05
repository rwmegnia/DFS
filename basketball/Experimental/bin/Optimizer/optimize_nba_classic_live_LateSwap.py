# -*- coding: utf-8 -*-
import pandas as pd
from pydfs_lineup_optimizer import Site, Sport, get_optimizer, PositionsStack, TeamStack
from pydfs_lineup_optimizer.fantasy_points_strategy import RandomFantasyPointsStrategy
import os
import warnings
from datetime import datetime
from ProcessRankings import processRankings
from optimizerTools import Locks, Exclusions, loadPlayers, mergeSwapProjectedPoints
import numpy as np

warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
game_date = datetime.now().strftime("%Y-%m-%d")
# game_date='2022-07-15'
datadir = f"{basedir}/../../data"
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NBA/Classic/{game_date}"
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
proj.rename({'team_abbreviation':'team'},axis=1,inplace=True)
proj=proj.merge(slate_salaries[['RotoName','team','Roster Position','ID','game_time']],
                on=['RotoName','team'],
                how='left')
proj['game_time']=pd.to_datetime(proj.game_time)
proj=proj[proj.ID.isna()==False]
proj['Projection']=proj[['RG_projection','DG_Proj','DG_Proj','ML','ML','RG_projection']].mean(axis=1)

#%%
# Create optimizerti
if contestType == "Showdown":
    proj.loc[proj["Roster Position"] == "CPT", "Projection"] *= 1.5
    optimizer = get_optimizer(Site.DRAFTKINGS_CAPTAIN_MODE, Sport.BASKETBALL)
else:
    optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)

# Players to replace
#
# # UTIL Nurkic 6900 ID 25376281 Replace with  25376302
# proj.rename({'team':'team_abbreviation',
#               'salary':'Salary'},axis=1,inplace=True)
# players = loadPlayers(proj, 'Projection', "Classic")
# optimizer.player_pool.load_players(players)
proj.PMM.fillna(0,inplace=True)
proj['min_exposure']=0
proj['max_exposure']=1

optimizer.load_players_from_csv("/Users/robertmegnia/Downloads/DKEntries.csv")
proj['Projection']=proj[['RG_projection','RG_projection','DG_Proj','ML']].mean(axis=1)
players = [
    mergeSwapProjectedPoints(p, proj,'Projection')
    for p in optimizer.player_pool.all_players if 
    mergeSwapProjectedPoints(p, proj,'Projection') is not None
]
optimizer.player_pool.load_players(players)
lineups = optimizer.load_lineups_from_csv("/Users/robertmegnia/Downloads/DKEntries.csv")

# optimizer.player_pool.remove_player('Charlie Blackmon')

Exclude = [
            'Trae young'
            # 'Jamal Murray',
            # 'Bogdan Bogdanovic'
            ]
Exclusions(Exclude, optimizer)
# Include=['Tj Mcconnell',
#          'Kawhi Leonard']
# Locks(Include,optimizer)
new_lineups=set(optimizer.optimize_lineups([lineups[0]]))
frames=[]
optimizer.set_min_salary_cap(49700)
optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0,0.25))

for lineup in optimizer.optimize_lineups(lineups):
    print(lineup)
optimizer.export('/Users/robertmegnia/Desktop/NBALineups1.csv')
df=pd.read_csv('/Users/robertmegnia/Desktop/NBALineups1.csv')
lineups_to_update=2
frames=[]
for i in range(0,lineups_to_update):
    frames.append(df[i::lineups_to_update].sample(1))

df=pd.concat(frames)
df.to_csv('/Users/robertmegnia/Desktop/NBALineups1.csv')
