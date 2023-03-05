#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:59:54 2021

@author: robertmegnia
"""
import numpy as np
import pandas as pd
import nflfastpy as nfl
from scipy.stats import norm
import requests
import os

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f'{basedir}/../../etc'
datadir= f'{basedir}/../../data'
projdir= f'{basedir}/../../LiveProjections'
from config.Stochastic_config import *
from config.MLModel_config import *
from PMMPlayer import getPMM
from getDKPts import getDKPts
from RosterUtils import *
from TeamDepthChartModel import TeamDepthChartModelPredict
from RookieMLModelPredict import rookiePrediction
from PlayerSharesModel import PlayerSharesPrediction
from TeamStatsModel import TeamStatsModelPredict
# os.chdir(f'{basedir}/../')
from Models.ModelDict import *
from ScrapeBettingOdds import ScrapeBettingOdds
#%%

def rolling_average(df,window=8):
    return df.rolling(min_periods=1, window=window).mean()

def StochasticPrediction(player, salary, pos, stats_df, opp, weekly_proj_df, stat_type):
    if stat_type == "DST":
        STATS = DK_DST_stats
    else:
        STATS = DK_stats
    stats_df.sort_values(by="game_date", inplace=True)
    name = stats_df[stats_df.gsis_id == player].full_name.unique()
    n_games = 16
    feature_frame = pd.DataFrame({})
    opp_feature_frame = pd.DataFrame({})
    player_df = stats_df[stats_df.gsis_id == player][-n_games:]
    opp_df = stats_df[
        (stats_df.opp == opp)
        & (stats_df.position == pos)
        & ((stats_df.salary >= (salary - 1000)) & (stats_df.salary <= salary + 1000))
    ][-n_games:]
    if len(player_df) == 0:
        return pd.DataFrame(
            {
                "Floor": [np.nan],
                "Ceiling": [np.nan],
                "UpsideProb": [np.nan],
                "Stochastic": [np.nan],
                "UpsideScore": [np.nan],
            }
        )
    salary = weekly_proj_df[weekly_proj_df.gsis_id == player].salary
    for stat in STATS:
        mean = player_df[stat].mean()
        std = player_df[stat].std()
        stats = np.random.normal(loc=mean, scale=std, size=10000)
        feature_frame[stat] = stats
        opp_mean = opp_df[stat].mean()
        opp_std = opp_df[stat].std()
        opp_stats = np.random.normal(loc=opp_mean, scale=opp_std, size=10000)
        opp_feature_frame[stat] = stats
    feature_frame.fillna(0, inplace=True)
    feature_frame = feature_frame.mask(feature_frame.lt(0), 0)
    ProjectionsFrame = pd.DataFrame({})
    ProjectionsFrame["Stochastic"] = getDKPts(feature_frame, stat_type)
    Floor = round(ProjectionsFrame.Stochastic.quantile(0.15), 1)
    if Floor < 0:
        Floor = 0
    Ceiling = round(ProjectionsFrame.Stochastic.quantile(0.85), 1)
    Stochastic = round(ProjectionsFrame.Stochastic.mean(), 1)
    std = ProjectionsFrame.Stochastic.std()
    loc = ProjectionsFrame.Stochastic.mean()
    # Predict Upside Probability
    upside_model = pickle.load(
        open(f"{etcdir}/model_pickles/{pos}_upside_score_model.pkl", "rb")
    )
    if np.isnan(salary.values):
        UpsideScore = np.nan
        return pd.DataFrame(
            {
                "Floor": [np.nan],
                "Ceiling": [np.nan],
                "UpsideProb": [np.nan],
                "Stochastic": [np.nan],
                "UpsideScore": [np.nan],
            }
        )

    else:
        UpsideScore = round(upside_model.predict(salary[:, np.newaxis])[0], 1)
    UpsideProb = round(1 - norm(loc=loc, scale=std).cdf(UpsideScore), 2)
    print(name, Stochastic, UpsideScore, UpsideProb)
    ProjFrame = pd.DataFrame(
        {
            "Floor": [Floor],
            "Ceiling": [Ceiling],
            "UpsideProb": [UpsideProb],
            "Stochastic": [Stochastic],
            "UpsideScore": [UpsideScore],
            "UpsideProb":[UpsideProb],
        }
    )
    return ProjFrame

def MLPrediction(stats_df,weekly_proj_df,stat_type):
    # Columns that we know the values of going into a week
    KnownFeatures=['total_line',
                   'proj_team_score',
                   'spread_line',
                   'opp_Rank',
                   'Adj_opp_Rank',
                   'salary',
                   'depth_team',
                   'Floor',
                   'Ceiling',
                   'Stochastic',
                   'UpsideProb',]
    if stat_type=='Offense':
        NonFeatures.append('UpsideScore')
    # Sort stats database chronologically and filter down to active players for the week
    stats_df.sort_values(by='game_date',inplace=True)
    stats_df=stats_df[stats_df.gsis_id.isin(weekly_proj_df.gsis_id)]
    
    # Create features for ML prediction
    features=stats_df.drop(KnownFeatures,axis=1,errors='ignore').groupby('gsis_id').apply(lambda x: rolling_average(x))

    # Reinsert gsis_id, get last of rolling averages, insert known features
    features[['gsis_id','position']]=stats_df[['gsis_id','position']]
    features=features.groupby('gsis_id',as_index=False).last()
    
    opp_features=stats_df.groupby(['opp','position','season','week'],as_index=False).sum()
    opp_features=opp_features.groupby(['opp','position']).apply(lambda x: rolling_average(x)).drop(NonFeatures+KnownFeatures+TeamShareFeatures,axis=1,errors='ignore')
    # Reinsert opp column and get last rolling average
    opp_features[['opp','position']]=stats_df.groupby(['opp','position','season','week'],as_index=False).first()[['opp','position']]
    opp_features=opp_features.groupby(['opp','position']).last()
    # Add _allowed suffix for defense featurse
    opp_features=opp_features.add_suffix('_allowed')
    opp_features.reset_index(inplace=True)
    
    # Merge Offense and Defense features
    features.set_index('gsis_id',inplace=True)
    features=features.drop(KnownFeatures,axis=1,errors='ignore').join(weekly_proj_df[KnownFeatures+['gsis_id','opp']].set_index('gsis_id')).reset_index()
    features=features.merge(opp_features,on=['opp','position'],how='left')   
    features.rename({'DKPts':'avg_DKPts'},axis=1,inplace=True)
    ProjFrame=weekly_proj_df[NonFeatures]
    ProjFrame.set_index('gsis_id',inplace=True)
    ProjFrame=ProjFrame.join(features.set_index('gsis_id').drop(NonFeatures,axis=1,errors='ignore'))
    #ProjFrame.RosterPercent.fillna(-1,inplace=True)
    ProjFrame.dropna(inplace=True)
    for position in ProjFrame.position.unique():
        models=ModelDict[position]
        for method,model in models.items():
            ProjFrame.loc[ProjFrame.position==position,method]=model.predict(ProjFrame.loc[ProjFrame.position==position,features.drop(NonFeatures,axis=1,errors='ignore').columns])  
            ProjFrame.loc[(ProjFrame.position==position)&(ProjFrame[method]<0),method]=0
    ProjFrame.reset_index(inplace=True)
    return ProjFrame
#%%
# Make Live Projection
Slate='Full'
week=1
season=2022
weekly_proj_df=getWeeklyRosters(season,week)
weekly_proj_df['RosterPercent']=5
proj_frames=[]
#%%
# Get Stochastic/ML Projections
for stat_type in ['DST','Offense']:
    if stat_type=='DST':
        pos_frame=weekly_proj_df[weekly_proj_df.position=='DST']
        df=pd.read_csv(
            f"{datadir}/game_logs/Full/{stat_type}_Database_StochProjections.csv"
        )
    else:
        pos_frame=weekly_proj_df[weekly_proj_df.position!='DST']
        df=pd.read_csv(
            f"{datadir}/game_logs/Full/{stat_type}_Database_snapcounts.csv"
        )
    df.game_date=pd.to_datetime(df.game_date)
    game_date=df[(df.season==season)&(df.week==week)].game_date.min()
    if pd.isnull(game_date):
        game_date=getGameDate(week,season)
    stats_df=df[df.game_date<game_date]
    tempSal=stats_df.groupby('gsis_id').tail(5)[['salary','gsis_id']].groupby('gsis_id',as_index=False).mean()
    pos_frame=pos_frame.merge(tempSal,on='gsis_id',how='left')
    pos_frame['depth_team']=pos_frame.groupby(['position','team','season','week']).salary.apply(lambda x: x.rank(ascending=False,method='min'))
    proj_df=pd.concat([pos_frame,pd.concat([a for a in pos_frame.apply(lambda x: StochasticPrediction(x.gsis_id,x.salary,x.position,stats_df,x.opp,pos_frame[pos_frame.gsis_id==x.gsis_id],stat_type),axis=1)]).set_index(pos_frame.index)],axis=1)
    ml_proj_df=MLPrediction(stats_df,proj_df,stat_type)
    proj_df=proj_df.merge(ml_proj_df[['gsis_id','BR','EN','NN','RF','GB','Tweedie']],on='gsis_id',how='left')
    proj_frames.append(proj_df)
proj_df=pd.concat(proj_frames)
proj_df=getPMM(proj_df,season,week)
proj_df['DepthChart']=proj_df.groupby(['team','position']).PMM.apply(lambda x: x.rank(ascending=False,method='min'))

# Get Player Share Predictions
share_proj_df=pd.concat([weekly_proj_df,pd.concat([a for a in weekly_proj_df.apply(lambda x: PlayerSharesPrediction(x.gsis_id,x.position,stats_df,x.opp,weekly_proj_df[weekly_proj_df.gsis_id==x.gsis_id]),axis=1)]).set_index(weekly_proj_df.index)],axis=1)

# Get Rookie Predictions
games_played=stats_df.groupby('gsis_id',as_index=False).size().rename({'size':'games_played'},axis=1)
stats_df=stats_df.merge(games_played,on='gsis_id',how='left')
weekly_proj_df=weekly_proj_df.merge(games_played,on='gsis_id',how='left')
weekly_proj_df['salary']=3000
weekly_proj_df['depth_team']=3
rookie_df=rookiePrediction(weekly_proj_df.drop('games_played',axis=1),stats_df) 
rookie_df=getPMM(rookie_df,season,week)

# Get Depth Chart Predictions
dc_proj=TeamDepthChartModelPredict(week,season)

# Get TeamStats Predictions
team_stats_proj=TeamStatsModelPredict(week,season)
top_down=share_proj_df.merge(team_stats_proj[['team_fpts',
                                'pass_attempt',
                                'rush_attempt',
                                'passing_fpts',
                                'rushing_fpts','receiving_fpts','week','season','opp','team']]
                                ,on=['week','season','team'],how='left')
top_down['TopDown']=(top_down.proj_passing_DKPts_share*top_down.passing_fpts)+(top_down.proj_rushing_DKPts_share*top_down.rushing_fpts)+(top_down.proj_receiving_DKPts_share*top_down.receiving_fpts)
# Get Top Down Model Predictions 
top_down_proj_df.to_csv(f'{projdir}/{season}/{Slate}/Week{week}_Projections.csv',index=False)
#%%