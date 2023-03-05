# -*- coding: utf-8 -*-
import pandas as pd
from pydfs_lineup_optimizer import Site, Sport, get_optimizer, PositionsStack, TeamStack
from pydfs_lineup_optimizer.fantasy_points_strategy import RandomFantasyPointsStrategy
import os
import warnings
from datetime import datetime
from ProcessRankings import processRankings
from optimizerTools import Locks, Exclusions, loadPlayers
import numpy as np
from sklearn.metrics import r2_score
warnings.simplefilter("ignore")
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
game_date = datetime.now().strftime("%Y-%m-%d")
salary_dir=f"/Volumes/XDrive/DFS/DraftKingsSalaryMegaDatabase2/NBA/Classic/{game_date}"
method='Projection'
season = 2022

### Read in Projections
proj = pd.read_csv(
    f"{datadir}/Projections/RealTime/{season}/Classic/{game_date}/{game_date}_Projections.csv"
)
proj.drop(proj[(proj.starter==True)&(proj.proj_mins<28)&(proj.ownership_proj<15)].index,inplace=True)
proj.drop(proj[(proj.starter==False)&(proj.proj_mins<15)&(proj.ownership_proj<15)].index,inplace=True)
proj.loc[proj[method].isna()==True,method]=proj.loc[proj[method].isna()==True,'RG_projection']
proj.loc[proj['Ceiling'].isna()==True,'Ceiling']=proj.loc[proj['Ceiling'].isna()==True,'RG_projection']*1.25
proj.loc[proj['Floor'].isna()==True,'Floor']=proj.loc[proj['Floor'].isna()==True,'RG_projection']*0.75

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
slate_salaries.rename({'team':'team_abbreviation'},axis=1,inplace=True)
proj.drop(['ID','salary'],axis=1,inplace=True)
proj=proj.merge(slate_salaries[['RotoName','team_abbreviation','position','Roster Position','ID','game_time','Salary']],
                on=['RotoName','team_abbreviation','position'],
                how='left')

proj=proj[proj.RG_projection.isna()==False]
#%%
proj=proj[proj.ID.isna()==False]
proj['game_time']=pd.to_datetime(proj.game_time)
proj['Projection']=proj[['RG_projection','DG_Proj','ML','RG_projection']].mean(axis=1)

# Create optimizer
optimizer = get_optimizer(Site.DRAFTKINGS, Sport.BASKETBALL)
proj['min_exposure']=0
proj['max_exposure']=1
# proj.loc[proj.player_name=='Naji Marshall','min_exposure']=1.0



players = loadPlayers(proj, method, "Classic")
optimizer.player_pool.load_players(players.to_list())
optimizer.set_fantasy_points_strategy(RandomFantasyPointsStrategy(0,0.25))

# optimizer.set_players_from_one_team({'PHI': 2,
#                                       'IND':2})
# optimizer.player_pool.exclude_teams(['SAS'])
Exclude = [
            'Aleperen Sengun',
            'Robert Williams'
            ]
# Exclusions(Exclude, optimizer)
Include=['Jayson Tatum','Shaedon Sharpe']
Locks(Include,optimizer)
# optimizer.player_pool.exclude_teams(['OKC','CHI'])
# optimizer.player_pool.lock_player("Kevin Durant",'SF')
# optimizer.set_projected_ownership(0,23)
# optimizer.set_min_salary_cap(49700)
# optimizer.set_players_from_one_team({'LAL':2})
# Execute Optimizer
for lineup in optimizer.optimize(n=100):
    lineup.actual_fantasy_points_per_game_projection=np.nan
    print(lineup)
# slate = slate.split("_Projections.csv")[0]
optimizer.print_statistic()
optimizer.export("Lineups.csv")
df=pd.read_csv('Lineups.csv')
# df.sample(20).to_csv('/Users/robertmegnia/Desktop/NBALineups.csv')
