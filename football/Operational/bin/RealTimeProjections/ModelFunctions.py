#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 19:55:11 2022

@author: robertmegnia
"""
import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy
from config.ColumnMappings import *
from getDKPts import getDKPts
from RosterUtils import getSchedule, getGameDate
from sklearn.cluster import KMeans
#%%
basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{basedir}/../../LiveProjections"
os.chdir(f"{basedir}/../")
from Models.ModelDict import *


def rolling_average(df, window):
    return df.rolling(min_periods=window, window=window).mean()


def standardizeStats(df, DK_stats):
    averages = df.rolling(window=8, min_periods=4).mean()[DK_stats]
    stds = df.rolling(window=8, min_periods=4).mean()[DK_stats]
    df[DK_stats] = (df[DK_stats] - averages[DK_stats]) / stds[DK_stats]
    df.fillna(0, inplace=True)
    return df


def OffenseStochastic(weekly_proj_df, stats_df):
    # tempSal=stats_df.groupby('gsis_id').tail(5)[['salary','gsis_id']].groupby('gsis_id',as_index=False).mean()
    # Filter weekly_proj_df to offense and set index to PlayerID
    N_games=8
    DK_stats = DK_Stoch_stats
    weekly_proj_df.loc[weekly_proj_df.position=='FB','position']='RB'
    stats_df.loc[stats_df.position=='FB','position']='RB'

    offense = weekly_proj_df[weekly_proj_df.position != "DST"]
    # offense=offense.merge(tempSal,on='gsis_id',how='left')
    offense["depth_team"] = offense.groupby(
        ["position", "team", "season", "week"]
    ).salary.apply(lambda x: x.rank(ascending=False, method="first"))
    offense.set_index(["player_id","position","depth_team"], inplace=True)
    stats_df.set_index(["player_id","position","depth_team"], inplace=True)
    offense = offense.join(stats_df[DK_stats], lsuffix="_a")

    # Get each offense last 14 games
    offense = offense.groupby(["player_id","position"]).tail(N_games)

    # Scale Offense stats frame to z scores to get opponent influence
    # Get Offense stats form last 14 games
    scaled_offense = (
        stats_df[stats_df.game_date > "2015-01-01"]
        .groupby(["player_id","position"])
        .tail(N_games*2)
    )
    scaled_offense.sort_index(inplace=True)
    # Get batters average/standard dev stats over last 20 games
    scaled_averages = (
        scaled_offense.groupby(["player_id","position"], as_index=False)
        .rolling(window=16, min_periods=1)
        .mean()[DK_stats]
        .groupby(["player_id","position"])
        .tail(N_games)
        .fillna(0)
    )
    scaled_stds = (
        scaled_offense.groupby(["player_id","position"], as_index=False)
        .rolling(window=16, min_periods=1)
        .std()[DK_stats]
        .groupby(["player_id","position"])
        .tail(N_games)
        .fillna(0)
    )
    scaled_offense = scaled_offense.groupby(["player_id","position"]).tail(N_games)
    # Get offense Z scores over last 7 games
    scaled_offense[DK_stats] = (
        (scaled_offense[DK_stats] - scaled_averages[DK_stats])
        / scaled_stds[DK_stats]
    ).values
    opp_stats = (
        scaled_offense.groupby(["opp", "game_date","position"])
        .mean()
        .groupby(["opp","position"])
        .tail(N_games)[DK_stats]
        .reset_index()
    )
    opp_stats = (
        opp_stats[opp_stats.opp.isin(weekly_proj_df.opp)].groupby(["opp",'position']).mean()
    )
    quantiles = scipy.stats.norm.cdf(opp_stats)
    quantiles = pd.DataFrame(quantiles, columns=DK_stats).set_index(
        opp_stats.index
    )
    #%%
    offense['sample_size']=offense.groupby(offense.index).size()
    offense=offense[offense.sample_size>1]
    offense.drop('sample_size',axis=1,inplace=True)
    averages = offense.groupby(["player_id","depth_team", "opp", "position"]).mean()
    averages = averages.reset_index().set_index(["opp", "position"])
    stds = offense.groupby(["player_id", "depth_team","opp", "position"]).std()
    stds = stds.reset_index().set_index(["opp", "position"])
    stds.fillna(0,inplace=True)
    quantiles = averages.join(quantiles[DK_stats], lsuffix="_quant")[DK_stats]
    quantiles.fillna(0, inplace=True)
    averages.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    quantiles.sort_index(inplace=True)
    shape = (averages[DK_Stoch_stats]/stds[DK_Stoch_stats])**2
    scale = (stds[DK_Stoch_stats]**2)/averages[DK_Stoch_stats]
    scale[scale<0]=0

    sims = np.random.gamma(
        shape=shape,
        scale=scale,
        size=(10000, len(averages), len(DK_Stoch_stats)),
    )
    sims[sims == np.nan] = 0
    offense = weekly_proj_df[weekly_proj_df.position != "DST"]
    # offense=offense.merge(tempSal,on='gsis_id',how='left')
    offense["depth_team"] = offense.groupby(
        ["position", "team", "season", "week"]
    ).salary.apply(lambda x: x.rank(ascending=False, method="min"))
    low = pd.DataFrame(
        np.quantile(sims, 0.1, axis=0), columns=DK_stats
    ).set_index(averages.player_id)
    low.rename({"DKPts": "Floor1"}, axis=1, inplace=True)
    low["Floor"] = getDKPts(low, "Offense")
    low["Floor"] = low[["Floor", "Floor1"]].mean(axis=1)
    
    high = pd.DataFrame(
        np.quantile(sims, 0.9, axis=0), columns=DK_stats
    ).set_index(averages.player_id)
    high["Ceiling"] = getDKPts(high, "Offense")
    high["Ceiling"] = high[["Ceiling", "DKPts"]].mean(axis=1)
    median = pd.DataFrame(
        np.quantile(sims, 0.5, axis=0), columns=DK_stats
    ).set_index(averages.player_id)
    median["Median"] = getDKPts(median, "Offense")
    median["Median"] = median[["Median", "DKPts"]].mean(axis=1)
    stoch = pd.concat(
        [
            pd.DataFrame(
                np.diag(
                    pd.DataFrame(sims[:, i, :], columns=DK_stats).quantile(
                        quantiles.values[i, :]
                    )
                ).reshape(1, -1),
                columns=DK_stats,
            )
            for i in range(0, len(averages))
        ]
    ).set_index(averages.player_id)
    stoch.rename({"DKPts": "Stochastic1"}, axis=1, inplace=True)
    stoch["Stochastic"] = getDKPts(stoch, "Offense")
    stoch["Stochastic"] = stoch[["Stochastic", "Stochastic1"]].mean(axis=1)
    offense.set_index("player_id", inplace=True)
    offense = offense.join(low["Floor"].round(1))
    offense = offense.join(high["Ceiling"].round(1))
    offense = offense.join(median["Median"].round(1))
    offense = offense.join(stoch["Stochastic"].round(1))
    low=low.add_suffix('_floor')
    median=median.add_suffix('_median')
    high=high.add_suffix('_ceiling')
    stoch = stoch.add_suffix('_stoch')
    offense = offense.join(low[[f'{c}_floor' for c in DK_stats if c!='DKPts']])
    offense = offense.join(median[[f'{c}_median' for c in DK_stats if c!='DKPts']])
    offense = offense.join(high[[f'{c}_ceiling' for c in DK_stats if c!='DKPts']])
    offense = offense.join(stoch[[f'{c}_stoch' for c in DK_stats if c!='DKPts']])
    offense['Stochastic']=offense[['Stochastic','Median']].mean(axis=1)

    offense.loc[
        (offense.position == "QB") & (offense.depth_team > 1), "Stochastic"
    ] = 0
    pos_frames = []
    stoch_nans=offense[offense.Stochastic.isna()==True]
    for pos in offense.position.unique():
        pos_frame = offense[(offense.position == pos)&(offense.Stochastic.isna()==False)]
        pos_frame.fillna(0, inplace=True)
        upside_model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_upside_score_model.pkl", "rb")
        )
        pos_frame["UpsideScore"] = upside_model.predict(
            pos_frame.salary[:, np.newaxis]
        )
        pos_frames.append(pos_frame)
    pos_frames.append(stoch_nans)
    offense = pd.concat(pos_frames)
    offense["std"] = (offense.Ceiling - offense.Floor) / 2
    offense["UpsideProb"] = 1 - norm.cdf(
        offense.UpsideScore, loc=offense.Stochastic, scale=offense["std"]
    )
    offense.UpsideProb = offense.UpsideProb.round(2)
    offense.drop("std", axis=1, inplace=True)
    offense.UpsideProb.fillna(0, inplace=True)
    offense.sort_values(by='Stochastic',ascending=False)
    offense.reset_index(inplace=True)
    offense["DepthChart"] = (
        offense.reset_index(drop=True)
        .groupby(["team", "position"])
        .salary.apply(lambda x: x.rank(ascending=False, method="first"))
    )
    return offense

def QBStochastic(players, stats_df):
    N_games=8
    players=players[players.position=='QB']
    stats_df=stats_df[stats_df.position=='QB']
    # Filter Players to Pitchers and set index to PlayerID
    offense = players[players.position != "DST"]
    offense.loc[offense.position=='FB','position']='RB'
    offense['depth_team']=offense.groupby(['position','team','season','week']).salary.apply(lambda x: x.rank(ascending=False,method='min'))
    offense.set_index(["gsis_id",'position','depth_team'], inplace=True)
    stats_df.loc[stats_df.position=='FB','position']='RB'
    stats_df.set_index(["gsis_id",'position','depth_team'], inplace=True)

    offense = offense.join(stats_df[DK_Stoch_stats], lsuffix="_a")

    # Get each offense player last 16 games
    offense = offense.groupby(["gsis_id","position"]).tail(N_games)

    # Scale offense stats frame to z scores to get opponent influence
    # Get offense stats form last 14 games
    scaled_offense = (
        stats_df[stats_df.game_date>'2018-01-01']
        .groupby(["gsis_id","position"])
        .tail(N_games*2)
    )
    scaled_offense.sort_index(inplace=True)
    scaled_offense.reset_index(inplace=True)
    
    # Get playeraverage/standard dev stats over last N_games
    scaled_averages = (
        scaled_offense.groupby(["gsis_id",'position'])
        .rolling(window=N_games, min_periods=N_games)
        .mean()[DK_Stoch_stats].reset_index()
        .groupby(["gsis_id",'position'])
        .tail(N_games)
    )
    scaled_averages.drop('level_2',axis=1,inplace=True,errors='ignore')
    scaled_stds = (
        scaled_offense.groupby(["gsis_id",'position'])
        .rolling(window=N_games, min_periods=N_games)
        .std()[DK_Stoch_stats].reset_index()
        .groupby(["gsis_id",'position'])
        .tail(N_games)
    )
    scaled_stds.drop('level_2',axis=1,inplace=True,errors='ignore')

    scaled_offense = scaled_offense.groupby(["gsis_id","position"]).tail(N_games)
    # Get offense Z scores over last 7 games
    scaled_offense[DK_Stoch_stats] = (
        (scaled_offense[DK_Stoch_stats] - scaled_averages[DK_Stoch_stats])
        / scaled_stds[DK_Stoch_stats]
    ).values
    opp_stats = (
        scaled_offense.groupby(["opp", "game_date","position"])
        .mean()
        .groupby(["opp","position"])
        .tail(N_games)[DK_Stoch_stats]
        .reset_index()
    )
    opp_stats = opp_stats[opp_stats.opp.isin(players.opp)].groupby(["opp","position"]).mean()
    quantiles = scipy.stats.norm.cdf(opp_stats)
    quantiles = pd.DataFrame(quantiles, columns=DK_Stoch_stats).set_index(
        opp_stats.index
    )
    averages = offense.groupby(["gsis_id","depth_team", "opp","position"]).mean()
    averages = averages.reset_index().set_index(["opp","position"])
    stds = offense.groupby(["gsis_id","depth_team","opp","position"]).std()
    stds = stds.reset_index().set_index(["opp","position"])
    quantiles = averages.join(quantiles[DK_Stoch_stats], lsuffix="_quant")[
        DK_Stoch_stats
    ]
    quantiles.fillna(0.5, inplace=True)
    averages.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    quantiles.sort_index(inplace=True)
    sims = np.random.normal(
        loc=averages[DK_Stoch_stats],
        scale=stds[DK_Stoch_stats],
        size=(10000, len(averages), len(DK_Stoch_stats)),
    )
    # sims[sims<0]=0
    offense = players[players.position != "DST"]
    offense['depth_team']=offense.groupby(['position','team','season','week']).salary.apply(lambda x: x.rank(ascending=False,method='min'))
    offense.loc[offense.position=='FB','position']='RB'
    median = pd.DataFrame(
        np.quantile(sims, 0.5, axis=0), columns=DK_Stoch_stats
    ).set_index(averages.gsis_id)
    median.rename({"DKPts": "Median1"}, axis=1, inplace=True)
    median["Median"] = getDKPts(median, "Offense")
    median["Median"] = median[["Median","Median1"]].mean(axis=1)
    low = pd.DataFrame(
        np.quantile(sims, 0.1, axis=0), columns=DK_Stoch_stats
    ).set_index(averages.gsis_id)
    low.rename({"DKPts": "Floor1"}, axis=1, inplace=True)
    low["Floor"] = getDKPts(low, "Offense")
    low["Floor"] = low[["Floor", "Floor1"]].mean(axis=1)
    high = pd.DataFrame(
        np.quantile(sims, 0.9, axis=0), columns=DK_Stoch_stats
    ).set_index(averages.gsis_id)
    high["Ceiling"] = getDKPts(high, "Offense")
    high["Ceiling"] = high[["Ceiling", "DKPts"]].mean(axis=1)
    stoch = pd.concat(
        [
            pd.DataFrame(
                np.diag(
                    pd.DataFrame(
                        sims[:, i, :], columns=DK_Stoch_stats
                    ).quantile(quantiles.values[i, :])
                ).reshape(1, -1),
                columns=DK_Stoch_stats,
            )
            for i in range(0, len(averages))
        ]
    ).set_index(averages.gsis_id)
    stoch.rename({"DKPts": "Stochastic1"}, axis=1, inplace=True)
    stoch["Stochastic"] = getDKPts(stoch, "Offense")
    stoch["Stochastic"] = stoch[["Stochastic", "Stochastic1"]].mean(axis=1)
    offense.set_index("gsis_id", inplace=True)
    offense = offense.join(low["Floor"].round(1))
    offense = offense.join(high["Ceiling"].round(1))
    offense = offense.join(median["Median"].round(1))
    offense = offense.join(stoch["Stochastic"].round(1))
    offense['Median']=offense[['Median','Stochastic']].mean(axis=1).round(1)
    offense.reset_index(inplace=True)
    offense.loc[(offense.position=='QB')&(offense.depth_team>1),'Stochastic']=0
    pos_frames=[]
    stoch_nans=offense[offense.Stochastic.isna()==True]
    for pos in offense.position.unique():
        pos_frame=offense[(offense.Stochastic.isna()==False)&(offense.position==pos)]
        pos_frame.fillna(0,inplace=True)
        upside_model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_upside_score_model.pkl", "rb")
        )
        pos_frame['UpsideScore']=upside_model.predict(pos_frame.salary[:,np.newaxis])
        pos_frames.append(pos_frame)
    pos_frames.append(stoch_nans)
    offense=pd.concat(pos_frames)
    offense['std']=(offense.Ceiling-offense.Floor)/2
    offense['UpsideProb']=1 - norm.cdf(offense.UpsideScore,loc=offense.Stochastic,scale=offense['std'])
    offense.UpsideProb=offense.UpsideProb.round(2)
    offense.drop('std',axis=1,inplace=True)
    return offense
#%%
def RBStochastic(players, stats_df):
    N_games=16
    # Fit cluster model to determine matchup type for each player
    cluster_model=KMeans(n_clusters=4).fit(stats_df[['spread_line','total_line']])
    stats_df['matchup']=cluster_model.predict(stats_df[['spread_line','total_line']])
    offense = players[players.position != "DST"]
    offense['matchup']=cluster_model.predict(offense[['spread_line','total_line']])
    offense.loc[offense.position=='FB','position']='RB'
    offense['depth_team']=offense.groupby(['position','team','season','week']).salary.apply(lambda x: x.rank(ascending=False,method='first'))
    offense.set_index(["gsis_id",'position','depth_team','matchup'], inplace=True)
    stats_df.loc[stats_df.position=='FB','position']='RB'
    stats_df.set_index(["gsis_id",'position','depth_team','matchup'], inplace=True)

    offense = offense.join(stats_df[DK_Stoch_stats], lsuffix="_a")

    # Get each offense player last N games
    offense = offense.groupby(["gsis_id","position"]).tail(N_games)

    # Scale offense stats frame to z scores to get opponent influence
    # Get offense stats form last 14 games
    scaled_offense = (
        stats_df[stats_df.game_date>'2018-01-01']
        .groupby(["gsis_id","position"])
        .tail(N_games*2)
    )
    scaled_offense.sort_index(inplace=True)
    scaled_offense.reset_index(inplace=True)
    
    # Get playeraverage/standard dev stats over last N_games
    scaled_averages = (
        scaled_offense.groupby(["gsis_id",'position'])
        .rolling(window=N_games, min_periods=N_games)
        .mean()[DK_Stoch_stats].reset_index()
        .groupby(["gsis_id",'position'])
        .tail(N_games)
    )
    scaled_averages.drop('level_2',axis=1,inplace=True,errors='ignore')
    scaled_stds = (
        scaled_offense.groupby(["gsis_id",'position'])
        .rolling(window=N_games, min_periods=N_games)
        .std()[DK_Stoch_stats].reset_index()
        .groupby(["gsis_id",'position'])
        .tail(N_games)
    )
    scaled_stds.drop('level_2',axis=1,inplace=True,errors='ignore')

    scaled_offense = scaled_offense.groupby(["gsis_id","position"]).tail(N_games)
    scaled_offense[DK_Stoch_stats] = (
        (scaled_offense[DK_Stoch_stats] - scaled_averages[DK_Stoch_stats])
        / scaled_stds[DK_Stoch_stats]
    ).values
    opp_stats = (
        scaled_offense.groupby(["opp", "game_date","position"])
        .mean()
        .groupby(["opp","position"])
        .tail(N_games)[DK_Stoch_stats]
        .reset_index()
    )
    opp_stats = opp_stats[opp_stats.opp.isin(players.opp)].groupby(["opp","position"]).mean()
    quantiles = scipy.stats.norm.cdf(opp_stats)
    quantiles = pd.DataFrame(quantiles, columns=DK_Stoch_stats).set_index(
        opp_stats.index
    )
    averages = offense.groupby(["gsis_id","depth_team", "opp","position","matchup"]).mean()
    averages = averages.reset_index().set_index(["opp","position"])
    stds = offense.groupby(["gsis_id","depth_team","opp","position","matchup"]).std()
    stds = stds.reset_index().set_index(["opp","position"])
    quantiles = averages.join(quantiles[DK_Stoch_stats], lsuffix="_quant")[
        DK_Stoch_stats
    ]
    quantiles.fillna(0.5, inplace=True)
    averages.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    quantiles.sort_index(inplace=True)
    sims = np.random.normal(
        loc=averages[DK_Stoch_stats],
        scale=stds[DK_Stoch_stats],
        size=(10000, len(averages), len(DK_Stoch_stats)),
    )
    sims[sims<0]=0
    offense = players[players.position != "DST"]
    offense['depth_team']=offense.groupby(['position','team','season','week']).salary.apply(lambda x: x.rank(ascending=False,method='first'))
    offense.loc[offense.position=='FB','position']='RB'
    median = pd.DataFrame(
        np.quantile(sims, 0.5, axis=0), columns=DK_Stoch_stats
    ).set_index(averages.gsis_id)
    median.rename({"DKPts": "Median1"}, axis=1, inplace=True)
    median["Median"] = getDKPts(median, "Offense")
    median["Median"] = median[["Median","Median1"]].mean(axis=1)
    low = pd.DataFrame(
        np.quantile(sims, 0.1, axis=0), columns=DK_Stoch_stats
    ).set_index(averages.gsis_id)
    low.rename({"DKPts": "Floor1"}, axis=1, inplace=True)
    low["Floor"] = getDKPts(low, "Offense")
    low["Floor"] = low[["Floor", "Floor1"]].mean(axis=1)
    high = pd.DataFrame(
        np.quantile(sims, 0.9, axis=0), columns=DK_Stoch_stats
    ).set_index(averages.gsis_id)
    high["Ceiling"] = getDKPts(high, "Offense")
    high["Ceiling"] = high[["Ceiling", "DKPts"]].mean(axis=1)
    stoch = pd.concat(
        [
            pd.DataFrame(
                np.diag(
                    pd.DataFrame(
                        sims[:, i, :], columns=DK_Stoch_stats
                    ).quantile(quantiles.values[i, :])
                ).reshape(1, -1),
                columns=DK_Stoch_stats,
            )
            for i in range(0, len(averages))
        ]
    ).set_index(averages.gsis_id)
    stoch.rename({"DKPts": "Stochastic1"}, axis=1, inplace=True)
    stoch["Stochastic"] = getDKPts(stoch, "Offense")
    stoch["Stochastic"] = stoch[["Stochastic", "Stochastic1"]].mean(axis=1)
    offense.set_index("gsis_id", inplace=True)
    offense = offense.join(low["Floor"].round(1))
    offense = offense.join(high["Ceiling"].round(1))
    offense = offense.join(median["Median"].round(1))
    offense = offense.join(stoch["Stochastic"].round(1))
    offense['Median']=offense[['Median','Stochastic']].mean(axis=1).round(1)
    offense.reset_index(inplace=True)
    offense.loc[(offense.position=='QB')&(offense.depth_team>1),'Stochastic']=0
    pos_frames=[]
    stoch_nans=offense[offense.Stochastic.isna()==True]
    for pos in offense.position.unique():
        pos_frame=offense[(offense.Stochastic.isna()==False)&(offense.position==pos)]
        pos_frame.fillna(0,inplace=True)
        upside_model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_upside_score_model.pkl", "rb")
        )
        pos_frame['UpsideScore']=upside_model.predict(pos_frame.salary[:,np.newaxis])
        pos_frames.append(pos_frame)
    pos_frames.append(stoch_nans)
    offense=pd.concat(pos_frames)
    offense['std']=(offense.Ceiling-offense.Floor)/2
    offense['UpsideProb']=1 - norm.cdf(offense.UpsideScore,loc=offense.Stochastic,scale=offense['std'])
    offense.UpsideProb=offense.UpsideProb.round(2)
    offense.drop('std',axis=1,inplace=True)
    return offense
#%%
def DefenseStochastic(weekly_proj_df, stats_df):
    # dstTempSal=stats_df.groupby('gsis_id').tail(5)[['salary','gsis_id']].groupby('gsis_id',as_index=False).mean()
    # Filter weekly_proj_df to defense and set index to PlayerID
    N=8
    DK_DST_stats = DK_DST_Stoch_stats
    defense = weekly_proj_df[weekly_proj_df.position == "DST"]
    # defense=defense.merge(dstTempSal,on='gsis_id',how='left')
    defense["depth_team"] = defense.groupby(
        ["position", "team", "season", "week"]
    ).salary.apply(lambda x: x.rank(ascending=False, method="min"))
    defense.set_index(["team"], inplace=True)
    stats_df.set_index(["team"], inplace=True)
    defense = defense.join(stats_df[DK_DST_stats], lsuffix="_a")

    # Get each defense last 32 games
    defense = defense.groupby("team").tail(32)

    # Scale defense stats frame to z scores to get opponent influence
    # Get defense stats form last 14 games
    scaled_defense = (
        stats_df[stats_df.game_date > "2019-01-01"].groupby(["team"]).tail(32)
    )
    scaled_defense.sort_index(inplace=True)
    # Get batters average/standard dev stats over last 20 games
    scaled_averages = (
        scaled_defense.groupby(["team"], as_index=False)
        .rolling(window=16, min_periods=1)
        .mean()[DK_DST_stats]
        .groupby(["team"])
        .tail(N)
        .fillna(0)
    )
    scaled_stds = (
        scaled_defense.groupby(["team"], as_index=False)
        .rolling(window=16, min_periods=1)
        .std()[DK_DST_stats]
        .groupby(["team"])
        .tail(N)
        .fillna(0)
    )
    scaled_defense = scaled_defense.groupby(["team"]).tail(N)
    # Get defense Z scores over last 7 games
    scaled_defense[DK_DST_stats] = (
        (scaled_defense[DK_DST_stats] - scaled_averages[DK_DST_stats])
        / scaled_stds[DK_DST_stats]
    ).values
    opp_stats = (
        scaled_defense.groupby(["opp", "game_date"])
        .mean()
        .groupby(["opp"])
        .tail(N)[DK_DST_stats]
        .reset_index()
    )
    opp_stats = (
        opp_stats[opp_stats.opp.isin(weekly_proj_df.opp)].groupby("opp").mean()
    )
    quantiles = scipy.stats.norm.cdf(opp_stats)
    quantiles = pd.DataFrame(quantiles, columns=DK_DST_stats).set_index(
        opp_stats.index
    )
    #%%
    averages = defense.groupby(["team", "opp"]).mean().fillna(0)
    averages = averages.reset_index().set_index("opp")

    stds = defense.groupby(["team", "opp"]).std().fillna(0)
    stds = stds.reset_index().set_index("opp")
    quantiles = averages.join(quantiles[DK_DST_stats], lsuffix="_quant")[
        DK_DST_stats
    ]
    quantiles.fillna(0, inplace=True)
    averages.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    quantiles.sort_index(inplace=True)
    shape = (averages[DK_DST_stats]/stds[DK_DST_stats])**2
    scale = (stds[DK_DST_stats]**2)/averages[DK_DST_stats]
    scale[scale<0]=0
    # sims = np.random.normal(
    #     averages[DK_Stoch_stats],
    #     stds[DK_Stoch_stats],
    #     size=(10000, len(averages), len(DK_Stoch_stats)),
    # )
    sims = np.random.gamma(
        shape=shape,
        scale=scale,
        size=(10000, len(averages), len(DK_DST_stats)),
    )
    defense = weekly_proj_df[weekly_proj_df.position == "DST"]
    # defense=defense.merge(dstTempSal,on='gsis_id',how='left')
    defense["depth_team"] = defense.groupby(
        ["position", "team", "season", "week"]
    ).salary.apply(lambda x: x.rank(ascending=False, method="min"))
    low = pd.DataFrame(
        np.quantile(sims, 0.1, axis=0), columns=DK_DST_stats
    ).set_index(averages.team)
    low.rename({"DKPts": "Floor1"}, axis=1, inplace=True)
    low["Floor"] = getDKPts(low, "defense")
    low["Floor"] = low[["Floor", "Floor1"]].mean(axis=1)
    high = pd.DataFrame(
        np.quantile(sims, 0.9, axis=0), columns=DK_DST_stats
    ).set_index(averages.team)
    high["Ceiling"] = getDKPts(high, "defense")
    high["Ceiling"] = high[["Ceiling", "DKPts"]].mean(axis=1)
    median = pd.DataFrame(
        np.quantile(sims, 0.5, axis=0), columns=DK_DST_stats
    ).set_index(averages.team)
    median["Median"] = getDKPts(median, "defense")
    median["Median"] = median[["Median", "DKPts"]].mean(axis=1)
    stoch = pd.concat(
        [
            pd.DataFrame(
                np.diag(
                    pd.DataFrame(sims[:, i, :], columns=DK_DST_stats).quantile(
                        quantiles.values[i, :]
                    )
                ).reshape(1, -1),
                columns=DK_DST_stats,
            )
            for i in range(0, len(averages))
        ]
    ).set_index(averages.team)
    stoch.rename({"DKPts": "Stochastic1"}, axis=1, inplace=True)
    stoch["Stochastic"] = getDKPts(stoch, "defense")
    stoch["Stochastic"] = stoch[["Stochastic", "Stochastic1"]].mean(axis=1)
    defense.set_index("team", inplace=True)
    defense = defense.join(low["Floor"].round(1))
    defense = defense.join(high["Ceiling"].round(1))
    defense = defense.join(median["Median"].round(1))
    defense = defense.join(stoch["Stochastic"].round(1))
    defense['Stochastic']=defense[['Stochastic','Median']].mean(axis=1)
    upside_model = pickle.load(
        open(f"{etcdir}/model_pickles/DST_upside_score_model.pkl", "rb")
    )
    defense["UpsideScore"] = upside_model.predict(defense.salary[:, np.newaxis])
    loc = np.mean(sims[:, :, -1], axis=0)
    std = np.std(sims[:, :, -1], axis=0)
    defense["UpsideProb"] = 1 - norm(loc=loc, scale=std).cdf(
        defense.UpsideScore
    )
    defense.UpsideProb = defense.UpsideProb.round(2)
    defense.UpsideProb.fillna(0, inplace=True)
    defense["DepthChart"] = 1
    return defense


def MLPrediction(stats_df, weekly_proj_df, stat_type):
    # Columns that we know the values of going into a week
    if stat_type == "DST":
        NonFeatures = ML_DefenseNonFeatures
        DK_stats = DK_DST_Stoch_stats
        ml_feature_cols=ML_DEFENSE_FEATURE_COLS

    else:
        NonFeatures = ML_OffenseNonFeatures
        DK_stats = DK_Stoch_stats
        ml_feature_cols=ML_OFFENSE_FEATURE_COLS
    matchup_model=KMeans(n_clusters=4).fit(stats_df[['proj_team_score','total_line','spread_line']])     
    stats_df['matchup']=matchup_model.predict(stats_df[['proj_team_score','total_line','spread_line']])
    weekly_proj_df['matchup']=matchup_model.predict(weekly_proj_df[['proj_team_score','total_line','spread_line']])
    KnownFeatures = ML_KnownFeatures
    TeamShareFeatures = MLTeamShareFeatures
    # Get Opp Stats First
    opp_features = stats_df[stats_df.season>=2020].groupby("gsis_id").apply(
        lambda x: standardizeStats(x, DK_stats)
    )
    opp_features = (
        opp_features.groupby(
            ["opp", "season", "week", "depth_team", "position","matchup"]
        )
        .mean()
        .drop(
            NonFeatures + KnownFeatures + TeamShareFeatures,
            axis=1,
            errors="ignore",
        )
    )
    opp_features = opp_features.groupby(
        ["opp", "depth_team", "position","matchup"]
    ).apply(lambda x: rolling_average(x,window=5))
    opp_features = opp_features.groupby(
        ["opp", "position", "depth_team","matchup"]
    ).last()
    # Add _allowed suffix for defense featurse
    opp_features = opp_features.add_suffix("_allowed")
    opp_features.reset_index(inplace=True)
    # Sort stats database chronologically and filter down to active players for the week
    stats_df.sort_values(by="game_date", inplace=True)
    stats_df = stats_df[stats_df.gsis_id.isin(weekly_proj_df.gsis_id)]

    # Create features for ML prediction
    stats_df.set_index(['gsis_id','depth_team','position','matchup'],inplace=True)
    features = (
        stats_df.drop(KnownFeatures, axis=1, errors="ignore")
        .groupby(["gsis_id","depth_team",'position','matchup'])
        .apply(lambda x: rolling_average(x,5))
    )

    # Reinsert gsis_id, get last of rolling averages, insert known features
    # features[["gsis_id","depth_team", "position"]] = stats_df[["gsis_id","depth_team", "position"]]
    features.reset_index(inplace=True)
    features = features.groupby(["gsis_id","depth_team","position",'matchup'], as_index=False).last()


    # Merge Offense and Defense features
    features.set_index(["gsis_id","depth_team","position","matchup"], inplace=True)
    features = (
        features.drop(KnownFeatures, axis=1, errors="ignore")
        .join(
            weekly_proj_df[KnownFeatures + ["gsis_id", "position","opp"]].set_index(
                ["gsis_id","depth_team","position","matchup"]
            )
        )
        .reset_index()
    )
    features = features.merge(
        opp_features, on=["opp", "position", "depth_team","matchup"], how="left"
    )
    features.rename({"DKPts": "avg_DKPts"}, axis=1, inplace=True)
    ProjFrame = weekly_proj_df[
        [c for c in NonFeatures if c in weekly_proj_df.columns.to_list()]
    ]
    ProjFrame.set_index("gsis_id", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("gsis_id").drop(NonFeatures, axis=1, errors="ignore")
    )
    ML = ProjFrame.dropna()
    ML_nan = ProjFrame[ProjFrame.isna().any(axis=1)]
    ML_nan=ML_nan[~ML_nan.index.isin(ML.index)]
    ML_nan=ML_nan.drop(KnownFeatures,axis=1,errors='ignore').join(
        weekly_proj_df.set_index('gsis_id')[KnownFeatures])
    ML_nan=ML_nan.groupby(ML_nan.index).first()
    if stat_type!='DST':
        for position in ML.position.unique():
            for stat in ['pass_yards','pass_td','rush_yards','rush_td','rec','rec_yards','rec_td']:
                models=PlayerStatsModelDict[position][stat]
                for method,model in models.items():
                    if (position=='QB')&(stat in ['rec','rec_yards','rec_td']):
                        ML.loc[ML.position==position,f'{method}_{stat}']=0
                        continue
                    if (position in ['RB','WR','TE'])&(stat in ['pass_yards','pass_td']):
                        ML.loc[ML.position==position,f'{method}_{stat}']=0
                        continue
                    ML.loc[ML.position==position,f'{method}_{stat}']=model.predict(
                        ML.loc[
                            ML.position==position,
                            ml_feature_cols])  
                    ML.loc[(ML.position==position)&(ML[f'{method}_{stat}']<0),f'{method}_{stat}']=0
                ML.loc[ML.position==position,stat]=ML.loc[ML.position==position][[f'BR_{stat}',f'EN_{stat}',f'NN_{stat}',f'RF_{stat}',f'GB_{stat}',f'Tweedie_{stat}']].mean(axis=1)
    for position in ML.position.unique():
        models = ModelDict[position]
        for method, model in models.items():
            ML.loc[ML.position == position, method] = model.predict(
                ML.loc[
                    ML.position == position,
                    ml_feature_cols,
                ]
            )
            ML.loc[(ML.position == position) & (ML[method] < 0), method] = 0
    ML["ML"] = ML[["BR", "EN", "NN", "RF", "GB", "Tweedie"]].mean(axis=1)
    ProjFrame = pd.concat([ML, ML_nan])
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


def rookiePrediction(weekly_proj_df, stats_df):
    RookieNonFeatures = [
        "full_name",
        "gsis_id",
        "opp",
        "week",
        "position",
        "season",
        "rookie_year",
        "game_date",
        "team",
        "game_location",
        "game_day",
        "Slate",
    ]
    KnownFeatures = RookieKnownFeatures
    NonFeatures = RookieNonFeatures
    rookies1 = weekly_proj_df[~weekly_proj_df.gsis_id.isin(stats_df.gsis_id)]
    rookies2 = weekly_proj_df[
        weekly_proj_df.gsis_id.isin(stats_df[stats_df.games_played < 3].gsis_id)
    ]
    rookie_df = pd.concat([rookies1, rookies2]).drop_duplicates()
    weekly_proj_df = weekly_proj_df[
        weekly_proj_df.gsis_id.isin(rookie_df.gsis_id)
    ]
    # Sort stats database chronologically and filter down to teams that are playing this week
    stats_df.sort_values(by="game_date", inplace=True)

    # Create features for ML prediction
    features = (
        stats_df.drop(KnownFeatures, axis=1,errors='ignore')
        .groupby(["team", "season", "week", "opp"], as_index=False)
        .mean()
    )
    features = (
        features.groupby("team")
        .apply(lambda x: rolling_average(x,8))
        .drop(NonFeatures + ["games_played"], axis=1, errors="ignore")
    )
    # Reinsert team,season,week,opp, get last of rolling averages, insert known features
    features[
        ["team", "season", "week", "opp"] + KnownFeatures
    ] = stats_df.groupby(
        ["team", "season", "week", "opp"], as_index=False
    ).mean()[
        ["team", "season", "week", "opp"] + KnownFeatures
    ]
    features = features.groupby("team", as_index=False).last()

    opp_features = (
        stats_df.drop(KnownFeatures, axis=1)
        .groupby(["opp", "season", "week", "team"], as_index=False)
        .mean()
    )
    opp_features = (
        stats_df.groupby(["opp"])
        .apply(lambda x: rolling_average(x,5))
        .drop(NonFeatures + ["games_played"], axis=1, errors="ignore")
    )
    opp_features["opp"] = stats_df.groupby(
        ["opp", "season", "week", "team"], as_index=False
    ).mean()["opp"]
    opp_features = opp_features.groupby("opp").last()
    # Add _allowed suffix for defense featurse
    opp_features = opp_features.add_suffix("_allowed")
    opp_features.reset_index(inplace=True)
    # opp_features.salary_allowed.fillna(0,inplace=True)
    # Merge features with rookies data frame and drop duplicate columns
    features = features.merge(opp_features, on="opp", how="left")
    ProjFrame = weekly_proj_df[NonFeatures + KnownFeatures]
    ProjFrame = ProjFrame.merge(
        features.drop(
            KnownFeatures + ["week", "season", "opp", "DKPts", "RosterPercent"],
            axis=1,
            errors="ignore",
        ),
        on=["team"],
        how="left",
    )
    for position in rookie_df.position.unique():
        if position == "DST":
            continue
        models = RookieModelDict[position]
        features = KnownFeatures + RookieFeatures[position]
        upside_model = pickle.load(
            open(
                f"{etcdir}/model_pickles/{position}_upside_score_model.pkl",
                "rb",
            )
        )
        for method, model in models.items():
            ProjFrame.loc[
                ProjFrame.position == position, method
            ] = model.predict(
                ProjFrame.loc[ProjFrame.position == position, features]
            )
            ProjFrame.loc[ProjFrame[method] < 0, method] = 0
        ProjFrame.loc[
            ProjFrame.position == position, "UpsideScore"
        ] = upside_model.predict(
            ProjFrame.loc[ProjFrame.position == position, "salary"][
                :, np.newaxis
            ]
        )
    ProjFrame['ML']=ProjFrame[models.keys()].mean(axis=1)
    ProjFrame.set_index("gsis_id", inplace=True)
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


def PlayerSharesPrediction(stats_df, weekly_proj_df):
    KnownFeatures = PSM_KnownFeatures
    NonFeatures = PSM_NonFeatures
    # Sort Stats Database chronologically
    stats_df.sort_values(by="game_date", inplace=True)
    stats_df = stats_df[stats_df.gsis_id.isin(weekly_proj_df.gsis_id)]
    weekly_proj_df = weekly_proj_df[
        weekly_proj_df.gsis_id.isin(stats_df.gsis_id)
    ]
    # Retrieve player name and position for this iteration
    window = 8

    # Load in model dictionary
    # create a list of the feature columns to predict
    feature_cols = (
        PlayerShareModel_features_dict["rushing_DKPts_share"]
        + PlayerShareModel_features_dict["receiving_DKPts_share"]
    )
    feature_cols.remove('depth_team')

    feature_frame = stats_df.groupby(["gsis_id","depth_team","position"]).apply(
        lambda x: rolling_average(x, window)
    )[feature_cols]
    feature_frame[["gsis_id","depth_team","position"]] = stats_df[["gsis_id","depth_team","position"]]
    feature_frame = feature_frame.groupby(["gsis_id","depth_team","position"], as_index=False).last()
    feature_frame=weekly_proj_df[['gsis_id','position','depth_team']].merge(feature_frame,on=['gsis_id','depth_team',"position"],how='left')
    feature_frame = feature_frame.merge(
        weekly_proj_df[["gsis_id", "position","depth_team"]], on=["gsis_id","depth_team","position"], how="left"
    )
    #
    ProjFrame = weekly_proj_df[NonFeatures + KnownFeatures]
    ProjFrame = ProjFrame.merge(
        feature_frame.drop(KnownFeatures, axis=1, errors="ignore"),
        on=["gsis_id", "position"],
        how="left",
    )
    ProjFrame.dropna(inplace=True)
    pos_stats = []
    for pos in ShareStatsModelDict.keys():
        models = ShareStatsModelDict[pos]
        for stat in models.keys():
            for method, model in models[stat].items():
                pos_stats.append(stat)
                ProjFrame.loc[
                    ProjFrame.position == pos, f"{method}"
                ] = model.predict(
                    ProjFrame[ProjFrame.position == pos][
                        PlayerShareModel_features_dict[stat]
                    ]
                )
            ProjFrame.loc[ProjFrame.position == pos, stat] = ProjFrame.loc[
                ProjFrame.position == pos, models[stat].keys()
            ].mean(axis=1)
    ProjFrame.reset_index(drop=True, inplace=True)
    ProjFrame.loc[ProjFrame.position == "QB", "receiving_DKPts_share"] = 0
    ProjFrame.loc[
        (ProjFrame.position == "QB") & (ProjFrame.depth_team == 1),
        "passing_DKPts_share",
    ] = 1
    ProjFrame.passing_DKPts_share.fillna(0, inplace=True)
    return ProjFrame


def TeamDepthChartPrediction(off_db, schedule):
    KnownFeatures = TDM_KnownFeatures
    NonFeatures = TDM_NonFeatures
    off_db.sort_values(by="game_date", inplace=True)
    off_db = off_db[off_db.team.isin(schedule.team)]
    features = (
        off_db.groupby("team")
        .apply(lambda x: rolling_average(x, window=8))
        .drop(KnownFeatures + NonFeatures, errors="ignore", axis=1)
    )
    features[["team", "opp"]] = off_db[["team", "opp"]]
    features[KnownFeatures] = off_db[KnownFeatures]
    features = features.groupby("team").last()
    features.fillna(0, inplace=True)
    opp_features = off_db.groupby("opp").apply(
        lambda x: rolling_average(x, window=8).drop(
            KnownFeatures + NonFeatures, errors="ignore", axis=1
        )
    )
    opp_features = opp_features.add_suffix("_allowed")
    opp_features[["opp", "team"]] = off_db[["opp", "team"]]
    opp_features = opp_features.groupby("opp").last()
    opp_features.fillna(0, inplace=True)
    features = features.merge(opp_features, on="opp", how="left")
    ProjectionsFrame = schedule.set_index("team").drop(
        KnownFeatures, errors="ignore", axis=1
    )
    ProjectionsFrame.sort_index(inplace=True)
    features.sort_values(by="team", inplace=True)
    for dc_pos in TeamModelDict.keys():
        PosProjectionsFrame = pd.DataFrame({"team": ProjectionsFrame.index})
        for method in TeamModelDict[dc_pos].keys():
            model = TeamModelDict[dc_pos][method]
            PosProjectionsFrame[f"proj_{dc_pos}_{method}"] = model.predict(
                features[feature_cols].rename({dc_pos: f"avg_{dc_pos}"}, axis=1)
            )
        ProjectionsFrame[f"proj_{dc_pos}"] = PosProjectionsFrame.set_index(
            "team"
        ).mean(axis=1)
    return ProjectionsFrame


def TeamStatsPrediction(off_db, schedule):
    KnownFeatures = TSM_KnownFeatures
    NonFeatures = TSM_NonFeatures
    off_db.sort_values(by="game_date", inplace=True)
    off_db = off_db[off_db.team.isin(schedule.team)]
    features = (
        off_db.groupby("team")
        .apply(lambda x: rolling_average(x, window=8))
        .drop(KnownFeatures + NonFeatures, errors="ignore", axis=1)
    )
    features[["team", "opp"]] = off_db[["team", "opp"]]
    # features[KnownFeatures] = off_db[KnownFeatures]
    features = features.groupby("team").last()
    features.fillna(0, inplace=True)
    opp_features = off_db.groupby("opp").apply(
        lambda x: rolling_average(x, window=8).drop(
            KnownFeatures + NonFeatures, errors="ignore", axis=1
        )
    )
    opp_features = opp_features.add_suffix("_allowed")
    opp_features[["opp", "team"]] = off_db[["opp", "team"]]
    opp_features = opp_features.groupby("opp").last()
    opp_features.fillna(0, inplace=True)
    features = features.merge(opp_features, on="opp", how="left")
    features.set_index("team", inplace=True)
    schedule.set_index("team", inplace=True)
    schedule.sort_index(inplace=True)
    features = features.join(schedule[KnownFeatures])
    features.reset_index(inplace=True)
    features.sort_values(by="team", inplace=True)
    ProjectionsFrame = schedule
    ProjectionsFrame.sort_index(inplace=True)
    for stat in TeamStatsModel_features_dict.keys():
        feats = TeamStatsModel_features_dict[stat]
        MLProjectionsFrame = pd.DataFrame({})
        for method in TeamStatsModelDict[stat].keys():
            model = TeamStatsModelDict[stat][method]
            MLProjectionsFrame[f"{stat}_{method}"] = model.predict(
                features[feats]
            )
        ProjectionsFrame[stat] = MLProjectionsFrame.mean(axis=1).values
        features[stat] = MLProjectionsFrame.mean(axis=1)
    return ProjectionsFrame.reset_index()


def TeamDepthChartModelPredict(week, season, schedule, odds):
    schedule = schedule.merge(
        odds[["team", "proj_team_score", "total_line", "spread_line"]],
        on="team",
        how="left",
    )
    schedule.rename(
        {"total_line": "TotalPoints", "proj_team_score": "ImpliedPoints"},
        axis=1,
        inplace=True,
    )
    schedule["TotalPoints_allowed"] = schedule.TotalPoints
    schedule["ImpliedPoints_allowed"] = (
        schedule.TotalPoints - schedule.ImpliedPoints
    )
    schedule["total_line"] = schedule.TotalPoints
    game_date = getGameDate(week, season)
    off_db = pd.read_csv(f"{datadir}/TeamDatabase/TeamOffenseStats_DB.csv")
    off_db.game_date = pd.to_datetime(off_db.game_date)
    off_db = off_db[off_db.game_date < game_date]
    proj_df = TeamDepthChartPrediction(off_db, schedule)
    return proj_df.reset_index()


def TeamStatsModelPredict(week, season, schedule, odds):
    sdf = schedule.merge(
        odds[["team", "proj_team_score", "total_line", "spread_line"]],
        on="team",
        how="left",
    )
    sdf.rename({"proj_team_score": "ImpliedPoints"}, axis=1, inplace=True)
    sdf.rename({"total_line": "TotalPoints"}, axis=1, inplace=True)
    sdf["ImpliedPoints_allowed"] = sdf.TotalPoints - sdf.ImpliedPoints
    game_date = getGameDate(week, season)
    off_db = pd.read_csv(f"{datadir}/TeamDatabase/TeamOffenseStats_DB.csv")
    off_db.game_date = pd.to_datetime(off_db.game_date)
    off_db = off_db[off_db.game_date < game_date]
    proj_df = TeamStatsPrediction(off_db, sdf)
    proj_df.to_csv(
        f"{datadir}/Projections/TeamProjections/{season}_Week{week}_TeamProjections.csv",
        index=False,
    )
    return proj_df
