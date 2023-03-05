#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 22:04:16 2022

@author: robertmegnia
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 08:59:54 2021

@author: robertmegnia
"""
import numpy as np
import pandas as pd
import os

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
from config.MLModel_config import *
from config.StochasticModel_config import *
from getDKPts import getDKPts
import scipy
os.chdir("../")
from Models.ModelDict import *

os.chdir(f"{basedir}/../")

#%%
def rolling_average(df, window):
    return df.rolling(min_periods=3, window=window).mean()

def rolling_std(df, window):
    return df.rolling(min_periods=3, window=window).std()

def rolling_median(df, window):
    return df.rolling(min_periods=3, window=window).median()

def rolling_min(df, window):
    return df.rolling(min_periods=3, window=window).min()

def rolling_max(df, window):
    return df.rolling(min_periods=3, window=window).max()

def generateMinutesFeatures(database):
        frames=[]
        for window in [3,5]:
            print(window)
            Features = (
                database.groupby(["player_id","config"])
                .apply(lambda x: rolling_average(x, window))
                .drop(MinutesNonFeatures + MinutesKnownFeatures, axis=1, errors="ignore")
            )
            Features = Features.add_suffix(f'_{window}gm_mean')
            frames.append(Features)
        # for window in [3,5]:
        #     print(window)
        #     Features = (
        #         database.groupby(["player_id","config"])
        #         .apply(lambda x: rolling_std(x, window))
        #         .drop(MinutesNonFeatures + KnownFeatures, axis=1, errors="ignore")
        #     )
        #     Features = Features.add_suffix(f'_{window}gm_std')
        #     frames.append(Features)
        # for window in [3,5]:
        #     print(window)
        #     Features = (
        #         database.groupby(["player_id","config"])
        #         .apply(lambda x: rolling_median(x, window))
        #         .drop(MinutesNonFeatures + KnownFeatures, axis=1, errors="ignore")
        #     )
        #     Features = Features.add_suffix(f'_{window}gm_median')
        #     frames.append(Features)
        # for window in [3,5]:
        #     print(window)
        #     Features = (
        #         database.groupby(["player_id","config"])
        #         .apply(lambda x: rolling_min(x, window))
        #         .drop(MinutesNonFeatures + KnownFeatures, axis=1, errors="ignore")
        #     )
        #     Features = Features.add_suffix(f'_{window}gm_min')
        #     frames.append(Features)
        # for window in [3,5]:
        #     print(window)
        #     Features = (
        #         database.groupby(["player_id","config"])
        #         .apply(lambda x: rolling_max(x, window))
        #         .drop(MinutesNonFeatures + KnownFeatures, axis=1, errors="ignore")
        #     )
        #     Features = Features.add_suffix(f'_{window}gm_max')
        #     frames.append(Features)
        Features=pd.concat(frames,axis=1)
        Features['mins'] = database['mins']
        Features[KnownFeatures] = database[KnownFeatures]
        Features[MinutesNonFeatures] = database[MinutesNonFeatures].values
        return Features,Features.drop(MinutesNonFeatures + ['mins'], axis=1).columns.to_list()

def MLMinutesPrediction(stats_df,game_proj_df):
    start_df=[]
    stats_df['min']=0
    for starter in [True,False]:
        minute_stats=stats_df[stats_df.started==starter]
        players=game_proj_df[game_proj_df.starter==starter]
        minute_stats=minute_stats[minute_stats.player_id.isin(players.player_id)]
        if starter==True:
            minute_stats=minute_stats[minute_stats.mins>0]
            # minute_stats=minute_stats[minute_stats.mins>minute_stats.mins.quantile(0.01)]
        feats,feats_list=generateMinutesFeatures(minute_stats)
        feats=feats.groupby('player_id').last()
        # feats=feats[feats[feats_list].isna().any(axis=1)==False]
        feats=feats[feats_list].drop(['salary'],axis=1)
        feats.fillna(0,inplace=True)
        players.set_index('player_id',inplace=True)
        players.rename({'Salary':'salary'},axis=1,inplace=True)
        players=players.join(feats)
        nans=players[players[feats_list].isna().any(axis=1)==True]
        players=players[players[feats_list].isna().any(axis=1)==False]
        for method in MinsModelDict[str(starter)].keys():
            model=MinsModelDict[str(starter)][method]
            players[f'{method}_mins']=model.predict(players[feats_list])
        players['proj_mins']=players[[f'{method}_mins' for method in MinsModelDict[str(starter)].keys()]].mean(axis=1)
        start_df.append(players)
        start_df.append(nans)
    start_df=pd.concat(start_df)
    start_df.drop([feat for feat in feats_list if feat!='salary'],axis=1,inplace=True)
    return start_df

def MLPrediction(stats_df, game_proj_df):
    # Filter player down to active players for the week
    pos_frames=[]
    for pos in ['C','SG','PG','SF','PF']:
        opp_stats_df = stats_df[(stats_df.opp.isin(game_proj_df.opp))&(stats_df.position.str.contains(pos)==True)]
        try:
            pos_stats_df = stats_df[stats_df.player_id.isin(game_proj_df.player_id)]
        except AttributeError:
            game_proj_df.reset_index(inplace=True)
            pos_stats_df = stats_df[stats_df.player_id.isin(game_proj_df.player_id)]

        # Create prediction features frame by taking 15 game running average of player stats
        pos_stats_df.rename({'started':'starter'},axis=1,inplace=True)
        features = (
            pos_stats_df.groupby(["player_id","starter"])
            .apply(lambda x: rolling_average(x,window=5))
            .drop(NonFeatures + KnownFeatures, axis=1, errors="ignore")
        )
    
        # Reassign player_id,team to features frame
        features[["player_id", "team_abbreviation","starter"]] = pos_stats_df[
            ["player_id", "team_abbreviation","starter"]
        ]
        # if len(features) == 0:
        #     return
    
        # Get last row of features frame which is the latest running average for that player
        features = features.groupby(["player_id","starter"], as_index=False).last()
    
        opp_features = (
            opp_stats_df.groupby(["opp", "game_date"])
            .agg(DefenseFeatures)
        )
        opp_features = (
            opp_features.groupby("opp")
            .apply(lambda x: rolling_average(x,window=5))
            .add_suffix("_allowed")
        )
        opp_features = opp_features.groupby("opp").last()
    
        # Merge Offense and Defense features
        features.set_index(["player_id","starter"], inplace=True)
        features = features.join(
            game_proj_df[["player_id", "opp",'salary','avg_proj_mins','starter']].set_index(["player_id","starter"])
        ).reset_index()
        features = features.merge(opp_features, on=["opp"], how="left")
        features['mins']=features['avg_proj_mins']
        features.drop('avg_proj_mins',axis=1,inplace=True)
        # Setup projection frame
        game_proj_df.rename({"team": "team_abbreviation"}, axis=1, inplace=True)
        # Make projections
        for starter in ["starter", "bench"]:
            if starter == "starter":
                start = True
            else:
                start = False
            ProjFrame = game_proj_df.copy()
            ProjFrame.rename({'avg_proj_mins':'mins'},axis=1,inplace=True)
            # ProjFrame.loc[ProjFrame.starter==True,'starter']='starter'
            # ProjFrame.loc[ProjFrame.starter==False,'starter']='bench'
            ProjFrame=ProjFrame[(ProjFrame.position.str.contains(pos)==True)&(ProjFrame.starter==start)]
            ProjFrame.set_index(["player_id","starter"], inplace=True)
            ProjFrame = ProjFrame.join(
                features.set_index(["player_id","starter"]).drop(NonFeatures+KnownFeatures, axis=1, errors="ignore")
            )
            ProjFrame_nan=ProjFrame[ProjFrame[feats].isna().any(axis=1)==True]
            ProjFrame=ProjFrame[ProjFrame[feats].isna().any(axis=1)==False]
            ProjFrame.reset_index(inplace=True)
            for stat in ["pts","fg3m","ast","reb","stl","blk","to"]:
                    methods=['EN','RF','GB']
                    for method in methods:
                        model = StatsModelDict[method][pos][f"{starter}_{stat}"]
                        ProjFrame.loc[
                            ProjFrame.starter==start, f"proj_{stat}_{method}"
                        ] = model.predict(
                            ProjFrame.loc[
                                ProjFrame.starter == start,
                                feats,
                            ]
                        )
                        ProjFrame.loc[
                            (ProjFrame.starter==start)
                            & (ProjFrame[f"proj_{stat}_{method}"] < 0),
                            stat,
                        ] = 0
            models = ModelDict[starter][pos]
            for method, model in models.items():
                ProjFrame.loc[ProjFrame.starter==start,stat]=ProjFrame.loc[ProjFrame.starter==start,f'proj_{stat}_{method}']
                ProjFrame.loc[ProjFrame.starter == start, method] = model.predict(
                    ProjFrame.loc[
                        ProjFrame.starter == start,
                        feats,
                    ]
                )
                ProjFrame.loc[
                    (ProjFrame.starter == start) & (ProjFrame[method] < 0), method
                ] = 0
            ProjFrame.loc[ProjFrame.starter == start,"ML"] = ProjFrame.loc[ProjFrame.starter==start][models.keys()].mean(axis=1)
            ProjFrame.loc[ProjFrame.starter == start,stat] = ProjFrame.loc[ProjFrame.starter==start,f"{stat}"] = ProjFrame.loc[ProjFrame.starter==start][[f"proj_{stat}_{m}" for m in methods]].mean(axis=1)


            for stat in ["pts","fg3m","ast","reb","stl","blk","tov","dkpts"]:
                methods=['RF','GB']
                for method in methods:
                    model = SharesModelDict[method][pos][f"{starter}_{stat}"]
                    ProjFrame.loc[
                        ProjFrame.starter==start, f"proj_pct_{stat}_{method}"
                    ] = model.predict(
                        ProjFrame.loc[
                            ProjFrame.starter == start,
                            feats,
                        ]
                    )
                    ProjFrame.loc[
                        (ProjFrame.starter==start)
                        & (ProjFrame[f"proj_pct_{stat}_{method}"] < 0),
                        stat,
                    ] = 0
                ProjFrame.loc[ProjFrame.starter==start,f"proj_pct_{stat}"] = ProjFrame.loc[ProjFrame.starter==start][[f"proj_pct_{stat}_{m}" for m in methods]].mean(axis=1)
            pos_frames.append(ProjFrame)
        ProjFrame=pd.concat(pos_frames)
        ProjFrame_nan.reset_index(inplace=True)
    return pd.concat([ProjFrame,ProjFrame_nan])

#
def TeamStatsPredictions(team_games_df, team_stats_df):
    # Get 15 game running average of team statistics
    team_stats_df.drop(['moneyline','spread_line'],axis=1,inplace=True,errors='ignore')
    features = (
        team_stats_df.groupby(["team_abbreviation", "opp"])
        .apply(lambda x: rolling_average(x,window=5))
        .drop(TeamNonFeatures, axis=1, errors="ignore")
    )
    features[["team_abbreviation", "opp"]] = team_stats_df[["team_abbreviation", "opp"]]
    if len(features) == 0:
        return

    # Get last row of team running average statistics
    features = features.groupby(["team_abbreviation"], as_index=False).last()

    # Get 15 game running average of oppponent statistics and latest average
    opp_features = (
        team_stats_df.groupby(["opp", "game_date"])
        .sum()
        .drop(TeamNonFeatures+TeamKnownFeatures, axis=1, errors="ignore")
    )
    opp_features = (
        opp_features.groupby("opp")
        .apply(lambda x: rolling_average(x,window=5))
        .add_suffix("_allowed")
    )
    opp_features = opp_features.groupby("opp").last()

    # Merge Offense and Defense features
    features = features.merge(opp_features, on=["opp"], how="left")

    # Setup projection
    ProjFrame = team_games_df.set_index('team_abbreviation')
    ProjFrame = ProjFrame.join(
        features.set_index("team_abbreviation").drop(TeamNonFeatures+TeamKnownFeatures, axis=1, errors="ignore")
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame
    for stat in ["pts",'fg3m','ast','stl','blk','reb','to', "dkpts"]:
        methods=['RF','GB']
        for method in methods:
            model = TeamModelDict[method][f"{method}{stat}"]
            ProjFrame[f"{method}_proj_{stat}"] = model.predict(
                ProjFrame[features.drop(TeamNonFeatures, axis=1, errors="ignore").columns]
            )
            ProjFrame.loc[(ProjFrame[f"{method}_proj_{stat}"] < 0), f"{method}_proj_{stat}"] = 0
        ProjFrame[f'proj_{stat}']=ProjFrame[[f'{m}_proj_{stat}' for m in methods]].mean(axis=1)
    return ProjFrame




def StochasticPrediction(stats_df, game_proj_df):
    N_games=10
    DK_stats = DK_Stats+['dkpts','tripledouble','doubledouble']
    offense_nan=game_proj_df[game_proj_df.player_id.isna()==True]
    pos_frames=[]
    for pos in['C','PG','SG','PF','SF']:
        print(pos)
        offense=game_proj_df[(game_proj_df.player_id.isna()==False)&
                             (game_proj_df.RotoPosition.str.contains(pos)==True)]
        offense.set_index(["player_id",'starter','config'], inplace=True)
        pos_stats_df=stats_df[stats_df.position.str.contains(pos)==True]
        pos_stats_df.rename({'started':'starter'},axis=1,inplace=True)
        pos_stats_df.set_index(["player_id",'starter','config'], inplace=True)
        offense = offense.join(pos_stats_df[DK_stats], lsuffix="_a")
        # Get each offense last 15 games
        offense = offense.groupby(["player_id",
                                   "starter",
                                   "config"]).tail(N_games)
    
        # Scale Offense stats frame to z scores to get opponent influence
        # Get Offense stats form last 14 games
        scaled_offense = (
            pos_stats_df[pos_stats_df.game_date > "2015-01-01"]
            .groupby(["player_id","starter"])
            .tail(N_games*2)
        )
        scaled_offense.sort_index(inplace=True)
        scaled_averages = (
            scaled_offense.groupby(["player_id","starter"], as_index=False)
            .rolling(window=10, min_periods=10)
            .mean()[DK_stats]
            .groupby(["player_id","starter"])
            .tail(N_games)
            .fillna(0)
        )
        scaled_stds = (
            scaled_offense.groupby(["player_id","starter"], as_index=False)
            .rolling(window=10, min_periods=10)
            .std()[DK_stats]
            .groupby(["player_id","starter"])
            .tail(N_games)
            .fillna(0)
        )
        scaled_offense = scaled_offense.groupby(["player_id","starter"]).tail(N_games)
        frequency=scaled_offense.groupby(scaled_offense.index).size()
        frequency=frequency[frequency==10]
        scaled_offense=scaled_offense[scaled_offense.index.isin(frequency.index)]
        scaled_stds=scaled_stds[scaled_stds.index.isin(frequency.index)]
        scaled_averages=scaled_averages[scaled_averages.index.isin(frequency.index)]
    
        # Get offense Z scores over last 7 games
        scaled_offense[DK_stats] = (
            (scaled_offense[DK_stats] - scaled_averages[DK_stats])
            / scaled_stds[DK_stats]
        ).values
        opp_stats = (
            scaled_offense.groupby(["opp","starter","game_date"])
            .mean()
            .groupby(["opp","starter"])
            .tail(N_games)[DK_stats]
            .reset_index()
        )
        opp_stats.replace(np.inf,np.nan,inplace=True)
        opp_stats = (
            opp_stats[opp_stats.opp.isin(game_proj_df.opp)].groupby(["opp","starter"]).mean()
        )
        quantiles = scipy.stats.norm.cdf(opp_stats)
        quantiles = pd.DataFrame(quantiles, columns=DK_stats).set_index(
            opp_stats.index
        )
        #%%
        offense['sample_size']=offense.groupby(offense.index).size()
        offense.drop('sample_size',axis=1,inplace=True)
        averages = offense.groupby(["player_id","opp", "starter"]).mean()
        averages = averages.reset_index().set_index(["opp", "starter"])
        stds = offense.groupby(["player_id", "opp", "starter"]).std()
        stds = stds.reset_index().set_index(["opp", "starter"])
        stds.fillna(0,inplace=True)
        quantiles = averages.join(quantiles[DK_stats], lsuffix="_quant")[DK_stats]
        quantiles.fillna(0.5, inplace=True)
        averages.sort_index(inplace=True)
        stds.sort_index(inplace=True)
        quantiles.sort_index(inplace=True)
        # shape = (averages[DK_stats]/stds[DK_stats])**2
        # scale = (stds[DK_stats]**2)/averages[DK_stats]
        # scale[scale<0]=0
    
        sims = np.random.normal(
            loc=averages[DK_stats],
            scale=stds[DK_stats],
            size=(10000, len(averages), len(DK_stats)),
        )
        sims[sims == np.nan] = 0
        sims[sims < 0]=0
        offense = game_proj_df[(game_proj_df.player_id.isna()==False)&
                               (game_proj_df.RotoPosition.str.contains(pos)==True)]
        low = pd.DataFrame(
            np.quantile(sims, 0.1, axis=0), columns=DK_stats
        ).set_index(averages.player_id)
        low.rename({"dkpts": "Floor1"}, axis=1, inplace=True)
        low["Floor"] = getDKPts(low)
        low["Floor"] = low[["Floor", "Floor1"]].mean(axis=1)
        
        high = pd.DataFrame(
            np.quantile(sims, 0.9, axis=0), columns=DK_stats
        ).set_index(averages.player_id)
        high.rename({'dkpts':'Ceiling1'},axis=1,inplace=True)
        high["Ceiling"] = getDKPts(high)
        high["Ceiling"] = high[["Ceiling", "Ceiling1"]].mean(axis=1)
        median = pd.DataFrame(
            np.quantile(sims, 0.5, axis=0), columns=DK_stats
        ).set_index(averages.player_id)
        median.rename({'dkpts':'Median1'},axis=1,inplace=True)
        median["Median"] = getDKPts(median)
        median["Median"] = median[["Median", "Median1"]].mean(axis=1)
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
        stoch.rename({"dkpts": "Stochastic1"}, axis=1, inplace=True)
        stoch["Stochastic"] = getDKPts(stoch)
        stoch["Stochastic"] = stoch[["Stochastic", "Stochastic1"]].mean(axis=1)
        dd_idx=np.where(np.asarray(DK_stats)=='doubledouble')[0][0]
        td_idx=np.where(np.asarray(DK_stats)=='tripledouble')[0][0]
        dd = sims[:,:,dd_idx]
        dd = pd.DataFrame(dd,columns=averages.player_id).T
        dd[dd<1]=0
        dd[dd>1]=1
        dd=dd.sum(axis=1)/10000
        dd.name='DDProb'
        dd=dd.to_frame()
        td = sims[:,:,td_idx]
        td = pd.DataFrame(td,columns=averages.player_id).T
        td[td<1]=0
        td[td>1]=1
        td=td.sum(axis=1)/10000
        td.name='TDProb'
        td=td.to_frame()
        offense.set_index("player_id", inplace=True)
        offense = offense.join(low["Floor"].round(1))
        offense = offense.join(high["Ceiling"].round(1))
        offense = offense.join(median["Median"].round(1))
        offense = offense.join(stoch["Stochastic"].round(1))
        offense = offense.join(dd.round(2))
        offense = offense.join(td.round(2))
        pos_frames.append(offense)
    offense=pd.concat(pos_frames)
    offense_nan.set_index('player_id',inplace=True)
    offense=pd.concat([offense,offense_nan])
    # low=low.add_suffix('_floor')
    # median=median.add_suffix('_median')
    # high=high.add_suffix('_ceiling')
    # stoch = stoch.add_suffix('_stoch')
    # offense = offense.join(low[[f'{c}_floor' for c in DK_stats if c!='DKPts']])
    # offense = offense.join(median[[f'{c}_median' for c in DK_stats if c!='DKPts']])
    # offense = offense.join(high[[f'{c}_ceiling' for c in DK_stats if c!='DKPts']])
    # offense = offense.join(stoch[[f'{c}_stoch' for c in DK_stats if c!='DKPts']])
    # offense['Stochastic']=offense[['Stochastic','Median']].mean(axis=1)

    return offense

def TopDownPrediction(df):
    df["proj_pts"] = df[["proj_pts", "proj_team_score"]].mean(axis=1)
    df["TD_Proj"] = (
          (df.proj_pct_pts * df.proj_pts)
        + (df.proj_pct_fg3m * df.proj_fg3m * 0.5)
        + (df.proj_pct_ast * df.proj_ast * 1.5)
        + (df.proj_pct_reb * df.proj_reb * 1.25)
        + (df.proj_pct_stl * df.proj_stl * 2)
        + (df.proj_pct_blk * df.proj_blk * 2)
        - (df.proj_pct_tov * df.proj_to * 0.5)
    )
    df["TD_Proj2"] = df.proj_pct_dkpts * df.proj_dkpts
    return df

