#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:06:31 2023

@author: robertmegnia
"""
from datetime import datetime
from time import sleep
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
from sklearn.cluster import KMeans
data_dir='/Volumes/XDrive/DFS/basketball/Experimental/data'

#%% Define Functions
stat_dict={
    'Points':'pts',
    'Rebounds':'reb',
    'Assists':'ast',
    'Blocked Shots':'blk',
    'Steals':'stl',
    'Turnovers':'to',
    '3-PT Made':'fg3m',
    'Fantasy Score':'pp_fpts'}

def standardizeStats(df,stats,window=20):
        averages = df.rolling(window=window, min_periods=window).mean()[stats]
        stds = df.rolling(window=window, min_periods=window).mean()[stats]
        df[stats] = (df[stats] - averages[stats]) / stds[stats]
        df.fillna(0, inplace=True)
        return df
    
def statProbs(full_name,team,opp,prop,stat,position,total,spread,usg,mins):
    print(f'Running {stat} model for {full_name}')
    if stat not in stat_dict.keys():
        return
    stat=stat_dict[stat]
    player_df=player_db[(player_db.player_name==full_name)]
    X = player_df[['total_line','spread_line','usg_pct','mins']].dropna()
    cluster_model=KMeans(n_clusters=4).fit(X)
    player_df=player_df[player_df.index.isin(X.index)]
    player_df['matchup']=KMeans(n_clusters=4).fit_predict(X)
    matchup=cluster_model.predict([[total,spread,usg,mins]])[0]
    player_df=player_df[-82:]
    player_df=player_df[player_df.matchup==matchup][-25:]
    if len(player_df)<25:
        return
    else:
        window=len(player_df)
    player_df.rename({'pct_tov':'pct_to'},axis=1,inplace=True)
    team_df=team_db[(team_db.team_abbreviation==team)].tail(25)
    opp_df = standardized_team_db[standardized_team_db.opp==opp]
    opp_player_df = standardized_player_db[(standardized_player_db.opp==opp)&
                                           (standardized_player_db.position==position)].groupby('game_id').mean().tail(25)
    
    
    team_avg = team_df[f'team_{stat}'].mean()
    team_std = team_df[f'team_{stat}'].std()
    
    opp_avg_allowed = opp_df[f'team_{stat}'].mean()
    opp_avg_allowed_player = opp_player_df[stat].mean()
   
    # Player Average Points Share
    player_avg_share = player_df[f'pct_{stat}'].mean()   
    player_std_share = player_df[f'pct_{stat}'].std()   

    # Player Share Sims
    player_shape = (player_avg_share/player_std_share)**2
    player_scale = (player_std_share**2)/player_avg_share
    player_share_sims=np.random.gamma(shape=player_shape,scale=player_scale,size=10000)
    
    # Team Sims
    team_avg = team_avg+(team_std*opp_avg_allowed)
    team_shape = (team_avg/team_std)**2
    team_scale = (team_std**2)/team_avg
    team_sims = np.random.poisson(np.random.gamma(shape=team_shape,scale=team_scale,size=10000))
    
    # Projection from Player Share and Team Points frame
    frame1 = pd.DataFrame({f'team_{stat}':team_sims,
                           f'player_{stat}_share':player_share_sims})
    frame1[f'proj_{stat}']=(frame1[f'team_{stat}']*frame1[f'player_{stat}_share']).round(0)
    frame1['prop']=prop
    # Player Averages
    player_avg = player_df[stat].mean()  
    player_std = player_df[stat].std()  
    
    # Adjust player average for opponent
    player_avg=player_avg+(opp_avg_allowed_player*player_std)
    player_shape = (player_avg/player_std)**2
    player_scale = (player_std**2)/player_avg
    player_sims = np.random.poisson(np.random.gamma(shape=player_shape,scale=player_scale,size=10000))
    
    frame2= pd.DataFrame({f'proj_{stat}':player_sims})
    frame2['prop']=prop
    
    final = pd.concat([frame2])
    final.loc[final[f'proj_{stat}']>=final.prop,'Over']=1
    final.loc[final[f'proj_{stat}']<final.prop,'Over']=0
    overProb=final.Over.mean()
    stat_proj = final[f'proj_{stat}'].mean()
    if overProb>=0.50:
        recommendation='Over'
        OddsToHit=overProb
    else:
        recommendation='Under'
        OddsToHit=1-overProb

    final_frame =  pd.DataFrame({'full_name':[full_name],
                                 'team':[team],
                                 'stat':[stat],
                                 'prop':[prop],
                                 'proj':[stat_proj],
                                 'recommendation':[recommendation],
                                 'OddsToHit':[OddsToHit]
                                 })
    return final_frame

def blockedShotProbs(full_name,team,opp,prop):
    return
# Use Selenium to scrape NBA Player Props from PrizePicks

#initialize web driver
driver= webdriver.Chrome()
url = "https://app.prizepicks.com/"
driver.get(url)

#this is to get rid of the pop-up box that shows after you Selenium opens the prizepicks page
driver.find_element(by='xpath',value="/html/body/div[2]/div[3]/div/div/div[1]").click()

#Selecting NBA
driver.find_element(by='xpath',value="//div[@class='name'][normalize-space()= 'NBA']").click() 
sleep(1)

# Get available prop categories
categories = driver.find_element(By.CSS_SELECTOR,".stat-container").text.split('\n')

# Create a list for which each individual player prop will be stored
nbaPlayers=[]

# Iterate through each available category and then through each player prop
for category in categories:
    driver.find_element(By.XPATH,f"//div[text()='{category}']").click()
    projections = WebDriverWait(driver, 5).until(
      EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".projection")))
    prop_frames=[]
    for projection in projections:

        names = projection.find_element(by='xpath',value='.//div[@class="name"]').text
        proj= projection.find_element(by='xpath',value='.//div[@class="presale-score"]').get_attribute('innerHTML')
        print(names, category, proj)

        players = pd.DataFrame({
            'full_name': [names],
            f'{category}':[proj],
            }).set_index('full_name')
        players[f'{category}']=players[f'{category}'].astype(float)
        prop_frames.append(players)
    prop_frame=pd.concat(prop_frames)
    nbaPlayers.append(prop_frame)

# Concatenate props into one data frame
df=pd.concat(nbaPlayers)
df=df.groupby('full_name').max()
# Add date to data frame
game_date=datetime.now().strftime('%Y-%m-%d')
df['game_date']=game_date
df.to_csv(
    f'/Volumes/XDrive/DFS/basketball/Experimental/data/PrizePicksProps/NBA_Props_{game_date}.csv',
    )
#%%
player_db = pd.read_csv(
    '/Volumes/XDrive/DFS/basketball/Experimental/data/game_logs/PlayerStatsDatabase.csv'
    )
player_db=player_db[player_db.mins!=0]
odds_db=pd.read_csv('/Volumes/XDrive/DFS/basketball/Experimental/data/Database/OddsDatabase.csv')
player_db=player_db.merge(
    odds_db[['game_date','team_abbreviation','total_line','spread_line']],
    on=['game_date','team_abbreviation'],
    how='left')

# Merge lineup config into player_db/team_db
player_db=player_db.groupby(['player_id','game_id'],as_index=False).first()
player_db['pp_fpts']=(
    (player_db.pts)+
    (player_db.reb*1.2)+
    (player_db.ast*1.5)+
    (player_db.blk*3)+
    (player_db.stl*3)-
    (player_db.to)
    )
player_db.sort_values(by='game_date',inplace=True)
standardized_player_db = player_db.groupby(
    'player_id'
    ).apply(
        lambda x: standardizeStats(
            x, ['reb','ast','pts','stl','blk','to','fg3m','pp_fpts']
            )
        )
team_db=pd.read_csv(    
    '/Volumes/XDrive/DFS/basketball/Experimental/data/game_logs/TeamStatsDatabase.csv'
)
team_db=team_db.merge(
    odds_db[['game_date','team_abbreviation','total_line','spread_line']],
    on=['game_date','team_abbreviation'],
    how='left')
team_db['team_pp_fpts']=(
    (team_db.team_pts)+
    (team_db.team_reb*1.2)+
    (team_db.team_ast*1.5)+
    (team_db.team_blk*3)+
    (team_db.team_stl*3)-
    (team_db.team_to)
    )
team_db.sort_values(by='game_date',inplace=True)
player_db=player_db.merge(team_db[['team_id','game_id','team_pp_fpts']],
                          on=['team_id','game_id'],how='left')
player_db['pct_pp_fpts']=player_db.pp_fpts/player_db.team_pp_fpts
standardized_team_db = team_db.groupby(
    'team_id'
    ).apply(
        lambda x: standardizeStats(
            x, ['team_reb','team_ast','team_pts','team_stl',
                'team_blk','team_to','team_fg3m','team_pp_fpts']
            )
        )
#%%
players = player_db[player_db.player_name.isin(df.index)].groupby('player_name').last()[['team_abbreviation','position']]
players = players[players.position!='0']
players=players.join(df)
player_ids=player_db.groupby(['rotoname','player_id'],as_index=False).last()[['rotoname','player_id']].set_index('rotoname')

# starting_lineups = scrapeStartingLineups(game_date).set_index('RotoName')
starting_lineups=pd.read_csv(f'/Volumes/XDrive/DFS/basketball/Experimental/data/Projections/RealTime/2022/Classic/{game_date}/{game_date}_Projections.csv')

starting_lineups=starting_lineups[['full_name',
                                   'opp',
                                   'starter',
                                   'total_line',
                                   'moneyline',
                                   'spread_line',
                                   'proj_usg',
                                   'DG_proj_mins']].set_index(['full_name'])
starting_lineups.opp.replace({'PHO':'PHX'},inplace=True)
players=players.join(starting_lineups)
players=players[players.opp.isna()==False]
prop_frames=[]
for category in categories:
    if category not in stat_dict.keys():
        continue
    temp=players[players[category].isna()==False]
    for row in temp.iterrows():
        full_name=row[0]
        team=row[1]['team_abbreviation']
        opp=row[1]['opp']
        pos=row[1]['position']
        prop=row[1][category]
        total=row[1]['total_line']
        spread=row[1]['spread_line']
        usg=row[1]['proj_usg']
        mins=row[1]['DG_proj_mins']
        if (np.isnan(usg)==True)|(np.isnan(mins)==True):
            continue
        prop_frame=statProbs(full_name,team,opp,prop,category,pos,
                             total,spread,usg,mins)
        prop_frames.append(prop_frame)
props=pd.concat(prop_frames)
props.sort_values(by='OddsToHit',ascending=False,inplace=True)
props.to_csv(f'{data_dir}/PrizePicksProps/NBA_Props_{game_date}.csv',index=False)
#driver.close()
#%%
def verifyPrizePicksNBA(game_date,season):
    props = pd.read_csv(f'{data_dir}/PrizePicksProps/NBA_Props_{game_date}.csv')
    results = pd.read_csv(f'{data_dir}/game_logs/{season}/{game_date}/{game_date}_PlayerStats.csv')
    
    results.rename({'player_name':'full_name',
                    'team_abbreviation':'team'},
                   axis=1,
                   inplace=True)
    results['pp_fpts']=(
    (results.pts)+
    (results.reb*1.2)+
    (results.ast*1.5)+
    (results.blk*3)+
    (results.stl*3)-
    (results.to)
    )
    stats = list(props.stat.unique())
    props = props.merge(results[['full_name','team']+stats],
                        on=['full_name','team'],
                        how='left')
    for stat in props.stat.unique():
        props.loc[(props.stat==stat)&(props[stat]>props.prop),'result']='Over'
        props.loc[(props.stat==stat)&(props[stat]<props.prop),'result']='Under'
        props.loc[(props.stat==stat)&(props[stat]==props.prop),'result']='Push'
    
    props.loc[props.recommendation==props.result,'Hit']=1
    props.loc[props.recommendation!=props.result,'Hit']=0
    return props