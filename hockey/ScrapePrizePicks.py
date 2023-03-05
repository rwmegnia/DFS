#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:06:31 2023

@author: robertmegnia
"""

from datetime import datetime
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
import requests
data_dir='/Volumes/XDrive/DFS/hockey/Experimental/data'
#%%
driver= webdriver.Chrome()
# Define Functions
def standardizeStats(df,stats):
        averages = df.rolling(window=20, min_periods=20).mean()[stats]
        stds = df.rolling(window=20, min_periods=20).mean()[stats]
        df[stats] = (df[stats] - averages[stats]) / stds[stats]
        df.fillna(0, inplace=True)
        return df
    
def shotsOnGoalProbs(full_name,team,opp,prop,season=2022):
    team_df = team_db[team_db.team==team][-15:]
    team_df[['avg_shots','avg_Corsi']]=team_df[['shots','Corsi']].mean()
    team_df[['std_shots','std_Corsi']]=team_df[['shots','Corsi']].std()
    team_df=team_df.tail(1)[['team','avg_shots','avg_Corsi','std_shots','std_Corsi']]
    team_df['opp']=opp
    # League Averages
    league_avg=team_db[team_db.season==season].mean()[['shots','Corsi']]
    opp_df = standardized_db[standardized_db.opp==opp][-15:]
    opp_df[['shots','Corsi']]=opp_df[['shots','Corsi']].mean()
    opp_df=opp_df.tail(1)[['shots','Corsi']]
    # opp_df/=league_avg
    opp_df = opp_df.add_suffix('_opp_allowed').reset_index(drop=True)
    opp_df['opp']=opp
    team_df=pd.concat([team_df.reset_index(drop=True),opp_df],axis=1)
    
    # Adjust team average using opponents allowed stats
    mean_shots = team_df.avg_shots+(team_df.std_shots*team_df.shots_opp_allowed)
    mean_corsi = team_df.avg_Corsi+(team_df.std_Corsi*team_df.Corsi_opp_allowed)
    
    player_df = skater_db[skater_db.full_name==full_name][['full_name','team','shots','Corsi','shots_share','Corsi_share']]
    player_df['shooting_efficiency']=player_df.shots/player_df.Corsi
    player_df[['avg_shots',
               'avg_Corsi',
               'avg_shots_share',
               'avg_Corsi_share',
               'avg_shooting_efficiency']]=player_df.rolling(
                   window=15,min_periods=15
                   ).mean()[['shots',
                             'Corsi',
                             'shots_share',
                             'Corsi_share',
                             'shooting_efficiency']]
    player_df[['std_shots',
               'std_Corsi',
               'std_shots_share',
               'std_Corsi_share',
               'std_shooting_efficiency']]=player_df.rolling(
                   window=15,min_periods=15
                   ).std()[['shots',
                            'Corsi',
                            'shots_share',
                            'Corsi_share',
                            'shooting_efficiency']]
    player_df['opp']=opp
    player_df=player_df.merge(opp_df,on='opp',how='left').tail(1)
    
    player_mean_shots = player_df.avg_shots + (player_df.std_shots*player_df.shots_opp_allowed)
    player_mean_shots_share = player_df.avg_shots_share

    player_mean_corsi = player_df.avg_Corsi + (player_df.std_Corsi*player_df.Corsi_opp_allowed)
    player_mean_Corsi_share = player_df.avg_Corsi_share
    
    # Get Sims
    shape = (mean_shots/team_df.std_shots)**2
    scale = (team_df.std_shots**2)/mean_shots
    team_shots = np.random.poisson(np.random.gamma(shape=shape,scale=scale,size=10000))
    shape = (mean_corsi/team_df.std_Corsi)**2
    scale = (team_df.std_Corsi**2)/mean_corsi
    team_Corsi =  np.random.poisson(np.random.gamma(shape=shape,scale=scale,size=10000))
    shape = (player_mean_shots/player_df.std_shots)**2
    scale = (player_df.std_shots**2)/player_mean_shots
    shots = np.random.poisson(np.random.gamma(shape=shape,scale=scale,size=10000))
    shape = (player_df.avg_shooting_efficiency/player_df.std_shooting_efficiency)**2
    scale = (player_df.std_shooting_efficiency**2)/player_df.avg_shooting_efficiency
    efficiency = np.random.gamma(shape=shape,scale=scale,size=10000)
    shape = (player_mean_corsi/player_df.std_Corsi)**2
    scale = (player_df.std_Corsi**2)/player_mean_corsi
    corsi = np.random.poisson(np.random.gamma(shape=shape,scale=scale,size=10000))
    shape = (player_mean_shots_share/player_df.std_shots_share)**2
    scale = (player_df.std_shots_share**2)/player_mean_shots_share
    shots_share = np.random.gamma(shape=shape,scale=scale,size=10000)
    shape = (player_mean_Corsi_share/player_df.std_Corsi_share)**2
    scale = (player_df.std_Corsi_share**2)/player_mean_Corsi_share
    corsi_share = np.random.gamma(shape=shape,scale=scale,size=10000)
    
    # Share frame
    share_frame = pd.DataFrame({'team_shots':team_shots,
                                'shots_share':shots_share})
    share_frame['shots']=share_frame.team_shots*share_frame.shots_share
    share_frame['prop']=prop
    # Corsi Share Frame
    corsi_share_frame = pd.DataFrame({'team_Corsi':team_Corsi,
                                      'corsi_share':corsi_share,
                                      'efficiency':efficiency})
    corsi_share_frame['shots']=(
        corsi_share_frame.team_Corsi*corsi_share_frame.corsi_share*corsi_share_frame.efficiency
        )
    corsi_share_frame['prop']=prop
    
    # Corsi Frame
    corsi_frame = pd.DataFrame({'corsi':corsi,
                                'efficiency':efficiency})
    corsi_frame['shots']=corsi_frame.corsi*corsi_frame.efficiency
    corsi_frame['prop']=prop
    # shot frame
    shot_frame = pd.DataFrame({'shots':shots})
    shot_frame['prop']=prop
    
    # Combine frames
    shots_prob_df = pd.concat([share_frame,corsi_share_frame,
                               corsi_frame,shot_frame])
    shots_prob_df.loc[shots_prob_df.shots>=shots_prob_df.prop,'Over']=1
    shots_prob_df.loc[shots_prob_df.shots<shots_prob_df.prop,'Over']=0
    
    over_prob = shots_prob_df.Over.mean()
    proj_shots = shots_prob_df.shots.mean().round(2)
    if over_prob>.50:
        recommendation='Over'
        odds_to_hit=over_prob
    else:
        recommendation='Under'
        odds_to_hit=1-over_prob
    
    final_df=pd.DataFrame({'full_name':[full_name],
                           'team':[team],
                           'opp':[opp],
                           'prop':[prop],
                           'proj_shots':[proj_shots],
                           'recommendation':[recommendation],
                           'OddsToHit':[odds_to_hit]})
    return final_df

def blockedShotProbs(full_name,team,opp,prop):
    return
#%%
############## PRIZEPICKS ################################################
#

game_date=datetime.now().strftime('%Y-%m-%d')
url = "https://app.prizepicks.com/"

driver.get(url)

#this is to get rid of the pop-up box that shows after you Selenium opens the prizepicks page
driver.find_element(by='xpath',value="/html/body/div[2]/div[3]/div/div/div[1]").click()
#Selecting NHL
driver.find_element(by='xpath',value="//div[@class='name'][normalize-space()= 'NHL']").click() 
sleep(1)
#%%
categories = driver.find_element(By.CSS_SELECTOR,".stat-container").text.split('\n')
nhlPlayers=[]
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
    nhlPlayers.append(prop_frame)

df=pd.concat(nhlPlayers)
df=df.groupby('full_name').max()


#%%
df.to_csv(
    f'/Volumes/XDrive/DFS/hockey/Experimental/data/PrizePicksProps/NHL_Props_{game_date}.csv',
      index=False
    )
#%% Read in Projections
# probs=pd.read_csv(f'/Volumes/XDrive/DFS/hockey/Experimental/data/Projections/RealTime/2022/Classic/{game_date}/{game_date}_Projections.csv')
# probs=probs.merge(df,on=['full_name','game_date'],how='left')
# prop_dict={
#     0.5:'ProbHalfShot',
#     1.0:'Prob1Shot',
#     1.5:'Prob15Shot',
#     2.0:'Prob2Shot',
#     2.5:'Prob25Shot',
#     3.0:'Prob3Shot',
#     3.5:'Prob35Shot',
#     4.0:'Prob4Shot',
#     4.5:'Prob45Shot'
#     }
# probs['OverProb']=probs.apply(lambda x: x[prop_dict[x.shots_prop]] if np.isnan(x.shots_prop)==False else np.nan,axis=1)
# probs.loc[probs.OverProb<=.40,'Bet']='Under'
# probs.loc[probs.OverProb>=.60,'Bet']='Over'
# probs.loc[probs.Bet=='Under','OddsToHit']=1-probs.loc[probs.Bet=='Under','OverProb']
# probs.loc[probs.Bet=='Over','OddsToHit']=probs.loc[probs.Bet=='Over','OverProb']
# probs.sort_values(by='OddsToHit',ascending=False,inplace=True)
teamAbbrevsDict = {
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Montréal Canadiens": "MTL",
    "Ottawa Senators": "OTT",
    "Toronto Maple Leafs": "TOR",
    "Carolina Hurricanes": "CAR",
    "Florida Panthers": "FLA",
    "Tampa Bay Lightning": "TBL",
    "Washington Capitals": "WSH",
    "Chicago Blackhawks": "CHI",
    "Detroit Red Wings": "DET",
    "Nashville Predators": "NSH",
    "St. Louis Blues": "STL",
    "Calgary Flames": "CGY",
    "Colorado Avalanche": "COL",
    "Edmonton Oilers": "EDM",
    "Vancouver Canucks": "VAN",
    "Anaheim Ducks": "ANA",
    "Dallas Stars": "DAL",
    "Los Angeles Kings": "LAK",
    "San Jose Sharks": "SJS",
    "Columbus Blue Jackets": "CBJ",
    "Minnesota Wild": "MIN",
    "Winnipeg Jets": "WPG",
    "Arizona Coyotes": "ARI",
    "Vegas Golden Knights": "VGK",
    "Seattle Kraken": "SEA",
    "Phoenix Coyotes": "ARI",
}
data_dir='/Volumes/XDrive/DFS/hockey/Experimental/data'

# Get Available Player Prop Players from Database
skater_db = pd.read_csv(f'{data_dir}/game_logs/SkaterStatsDatabase.csv')
team_db = pd.read_csv(f'{data_dir}/game_logs/TeamStatsDatabase.csv')
team_db['shot_efficiency']=team_db.shots/team_db.Corsi
standardized_db = team_db.groupby("team").apply(
            lambda x: standardizeStats(x,['Corsi','shots','shot_efficiency'])
        )
# Create a dataframe of the available player prop players with their team and position
skaters = skater_db[skater_db.full_name.isin(df.index)].groupby('full_name',as_index=False).last()[['full_name','team','position']]

# We need to find out who the available players opponents will be by looking at
# their schedule. Get a list of each team

teams=skaters.team.unique()
    
# Get next game on schedule for every team
url = "https://statsapi.web.nhl.com/api/v1/teams?expand=team.schedule.next"
response = requests.get(url).json()

team_frames=[]
for team in response['teams']:
    team_abbrev = team["abbreviation"]
    if team_abbrev not in teams:
        continue
    team_name = team["name"]
    if (
        team_name
        == team["nextGameSchedule"]["dates"][0]["games"][0]["teams"]["away"][
            "team"
        ]["name"]
    ):
        game_location = "away"
        opp = team["nextGameSchedule"]["dates"][0]["games"][0]["teams"]["home"][
            "team"
        ]["name"]
        opp = teamAbbrevsDict[opp]
    else:
        game_location = "home"
        opp = team["nextGameSchedule"]["dates"][0]["games"][0]["teams"]["away"][
            "team"
        ]["name"]
        opp = teamAbbrevsDict[opp]
    team_df = pd.DataFrame({'team':[team_abbrev],
                            'opp':[opp]})
    team_frames.append(team_df)
team_df=pd.concat(team_frames)
#%% Merge team_df with skaters to obtain each players opponent
skaters=skaters.merge(team_df,on=['team'],how='left').set_index('full_name').join(df)
skaters.reset_index(inplace=True)

        
sog = skaters[skaters['Shots On Goal'].isna()==False]

sog_frames=[]
for row in sog.iterrows():
    full_name=row[1]['full_name']
    team=row[1]['team']
    opp=row[1]['opp']
    prop=row[1]['Shots On Goal']
    frame=shotsOnGoalProbs(full_name,team,opp,prop)
    sog_frames.append(frame)
sog_props=pd.concat(sog_frames)
sog_props.sort_values(by='OddsToHit',ascending=False,inplace=True)

sog_props.to_csv(f'{data_dir}/PrizePicksProps/NHL_SOG_Props_{game_date}.csv',index=False)
driver.close()
#%%
def verifyPrizePicksNHL(game_date,season):
    props = pd.read_csv(f'{data_dir}/PrizePicksProps/NHL_SOG_Props_{game_date}.csv')
    results = pd.read_csv(f'{data_dir}/game_logs/{season}/{game_date}/{game_date}_SkaterStats.csv')
    props = props.merge(results[['full_name','team','shots']],
                        on=['full_name','team'],
                        how='left')
    props.loc[props.shots>props.prop,'result']='Over'
    props.loc[props.shots<props.prop,'result']='Under'
    props.loc[props.shots==props.prop,'result']='Push'

    props.loc[props.recommendation==props.result,'Hit']=1
    props.loc[props.recommendation!=props.result,'Hit']=0
    props = props[props.shots.isna()==False]
    return props