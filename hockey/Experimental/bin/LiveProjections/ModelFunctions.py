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
import scipy
from NHL_API_TOOLS import *

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
from config.MLModel_config import *
from config.StochasticModel_config import *

os.chdir("../")
from Models.ModelDict import *

os.chdir(f"{basedir}/../")

#%%
def rolling_average(df, window=15):
    return df.rolling(min_periods=1, window=window).mean()

def skatersStochastic(players,stats_df):
    N_games=15
    players=players[players.position.isna()==False]
    players.ownership_proj.fillna(0, inplace=True)
    # Filter Players to skaters and set index to ID and to join with stats frame
    skaters = players[~players.position.isin(['G'])] 
    skaters = skaters[skaters.RotoPosition.isna() == False]
    skaters.set_index(["player_id",'position_type'], inplace=True)
    stats_df.set_index(["player_id",'position_type'], inplace=True)
    skaters = skaters.join(stats_df[DK_Skater_Stats+['DKPts']], lsuffix="_a")

    # Get each skaters last 20 games
    skaters = skaters.groupby(["player_id",'position_type']).tail(N_games+5)

    # Scale skater stats frame to z scores to get opponent influence
    # Get skater stats from last 20 games
    scaled_skaters = (
        stats_df[stats_df.game_date > "2021-01-1"]
        .groupby(["player_id",'position_type'])
        .tail(N_games+5)
    )
    scaled_skaters.sort_index(inplace=True)

    # Get skaters average/standard dev stats over last 20 games
    scaled_averages = (
        scaled_skaters.groupby(["player_id",'position_type'], as_index=False)
        .rolling(window=N_games, min_periods=5)
        .mean()[DK_Skater_Stats+['DKPts']]
        .groupby(["player_id",'position_type'])
        .tail(N_games)
    )

    scaled_stds = (
        scaled_skaters.groupby(["player_id",'position_type'], as_index=False)
        .rolling(window=N_games, min_periods=5)
        .std()[DK_Skater_Stats+['DKPts']]
        .groupby(["player_id",'position_type'])
        .tail(N_games)
    )

    scaled_skaters = scaled_skaters.groupby(["player_id",'position_type']).tail(N_games)

    # Get skaters Z scores over last 20 games
    scaled_skaters[DK_Skater_Stats+['DKPts']] = (
        (scaled_skaters[DK_Skater_Stats+['DKPts']] - scaled_averages[DK_Skater_Stats+['DKPts']])
        / scaled_stds[DK_Skater_Stats+['DKPts']]
    ).values
    #scaled_skaters.fillna(0,inplace=True)
    # Find out what opponents allowed to previous skaters over last
    opp_stats = (
        scaled_skaters.groupby(["opp", "game_date",'position_type'])
        .mean()
        .groupby(["opp",'position_type'])
        .tail(N_games)[DK_Skater_Stats+['DKPts']]
        .reset_index()
    )
    
    # Take averages of previous data frame
    opp_stats = (
        opp_stats[
            opp_stats.opp.isin(players.opp)
        ]
        .groupby(["opp",'position_type'])
        .mean()
    )
    opp_stats.fillna(0,inplace=True)
    #%%
    averages = skaters.groupby(["ID",'opp','position_type']).mean()
    averages = averages.reset_index().set_index(["opp",'position_type'])
    stds = skaters.groupby(["ID","opp",'position_type']).std()
    stds = stds.reset_index().set_index(['opp','position_type'])
    
    # Convert average Z Scores to quantiles
    #shape=(averages[DK_Skater_Stats+['DKPts']]/stds[DK_Skater_Stats+['DKPts']])**2
    #shape=opp_goalie_stats.join(shape,lsuffix='_shape')
    # quantiles = 1 - scipy.stats.gamma.sf(shape[[c for c in shape.columns if '_shape' not in c]],
    #                                      shape[[c for c in shape.columns if '_shape' in c]])
    quantiles = scipy.stats.norm.cdf(opp_stats)
    quantiles = pd.DataFrame(quantiles, columns=DK_Skater_Stats+['DKPts']).set_index(
        opp_stats.index
    )
    quantiles = averages.join(quantiles[DK_Skater_Stats+['DKPts']], lsuffix="_quant")[
        DK_Skater_Stats+['DKPts']
    ]
    quantiles.fillna(0.5, inplace=True)
    averages.sort_index(inplace=True)
    stds.sort_index(inplace=True)
    quantiles.sort_index(inplace=True)
    # averages.fillna(0,inplace=True)
    # sims = np.random.normal(
    #     averages[DK_Skater_Stats+['DKPts']],
    #     stds[DK_Skater_Stats+['DKPts']],
    #     size=(10000, len(averages), len(DK_Skater_Stats+['DKPts'])),
    # )
    sims = np.random.gamma(
        (averages[DK_Skater_Stats+['DKPts']]/stds[DK_Skater_Stats+['DKPts']])**2,
        (stds[DK_Skater_Stats+['DKPts']]**2)/averages[DK_Skater_Stats+['DKPts']],
        size=(10000, len(averages), len(DK_Skater_Stats+['DKPts'])),
    )
    # sims[sims < 0] = 0
    #
    skaters = players[~players.position.isin(['G'])] 
    skaters = skaters[skaters.RotoPosition.isna() == False]

    # Get Floor Stats
    low = pd.DataFrame(
        np.quantile(sims, 0.1, axis=0), columns=DK_Skater_Stats+['DKPts']
    ).set_index(averages.ID)
    low.rename({"DKPts": "Floor1"}, axis=1, inplace=True)
    low["Floor"] = getSkaterDKPts(low)
    low["Floor"] = low[["Floor", "Floor1"]].mean(axis=1)

    # Get Ceiling Stats
    high = pd.DataFrame(
        np.quantile(sims, 0.9, axis=0), columns=DK_Skater_Stats+['DKPts']
    ).set_index(averages.ID)
    high.rename({'DKPts':'Ceiling1'},axis=1,inplace=True)
    high["Ceiling"] = getSkaterDKPts(high)
    high["Ceiling"] = high[["Ceiling", "Ceiling1"]].mean(axis=1)

    # Get Stochastic Stats/Projections
    median = pd.concat(
        [
            pd.DataFrame(
                np.diag(
                    pd.DataFrame(
                        sims[:, i, :], columns=DK_Skater_Stats+['DKPts']
                    ).quantile(quantiles.values[i, :])
                ).reshape(1, -1),
                columns=DK_Skater_Stats+['DKPts'],
            )
            for i in range(0, len(skaters))
        ]
    ).set_index(averages.ID)
    median.rename({"DKPts": "Stochastic1"}, axis=1, inplace=True)
    median["Stochastic"] = getSkaterDKPts(median)
    median["Stochastic"] = median[["Stochastic", "Stochastic1"]].mean(axis=1)
    shot_idx=np.where(np.asarray(DK_Skater_Stats)=='shots')[0][0]
    goal_idx=np.where(np.asarray(DK_Skater_Stats)=='goals')[0][0]
    assist_idx=np.where(np.asarray(DK_Skater_Stats)=='assists')[0][0]

    HDSC_idx=np.where(np.asarray(DK_Skater_Stats)=='HDSC')[0][0]
    Block_idx=np.where(np.asarray(DK_Skater_Stats)=='blocked')[0][0]
    # Bonus Prob
    ShotBonusProb = sims[:,:,shot_idx].copy()
    ShotBonusProb = pd.DataFrame(ShotBonusProb,columns=averages.ID).T
    ShotBonusProb[ShotBonusProb<5]=0
    ShotBonusProb[ShotBonusProb>=5]=1
    ShotBonusProb=ShotBonusProb.sum(axis=1)/10000
    ShotBonusProb.name='ShotBonusProb'
    ShotBonusProb=ShotBonusProb.to_frame()
    # Half shot Prob
    ProbHalfShot = sims[:,:,shot_idx].copy()
    ProbHalfShot = pd.DataFrame(ProbHalfShot,columns=averages.ID).T
    ProbHalfShot[ProbHalfShot<0.5]=0
    ProbHalfShot[ProbHalfShot>=.5]=1
    ProbHalfShot=ProbHalfShot.sum(axis=1)/10000
    ProbHalfShot.name='ProbHalfShot'
    ProbHalfShot=ProbHalfShot.to_frame()
    # 1 shot
    Prob1Shot = sims[:,:,shot_idx].copy()
    Prob1Shot = pd.DataFrame(Prob1Shot,columns=averages.ID).T
    Prob1Shot[Prob1Shot<0.5]=0
    Prob1Shot[Prob1Shot>=.5]=1
    Prob1Shot=Prob1Shot.sum(axis=1)/10000
    Prob1Shot.name='Prob1Shot'
    Prob1Shot=Prob1Shot.to_frame()
    # 1.5 shot
    Prob15Shot = sims[:,:,shot_idx].copy()
    Prob15Shot = pd.DataFrame(Prob15Shot,columns=averages.ID).T
    Prob15Shot[Prob15Shot<1.5]=0
    Prob15Shot[Prob15Shot>=1.5]=1
    Prob15Shot=Prob15Shot.sum(axis=1)/10000
    Prob15Shot.name='Prob15Shot'
    Prob15Shot=Prob15Shot.to_frame()
    # 2 shots
    Prob2Shot = sims[:,:,shot_idx].copy()
    Prob2Shot = pd.DataFrame(Prob2Shot,columns=averages.ID).T
    Prob2Shot[Prob2Shot<2]=0
    Prob2Shot[Prob2Shot>=2]=1
    Prob2Shot=Prob2Shot.sum(axis=1)/10000
    Prob2Shot.name='Prob2Shot'
    Prob2Shot=Prob2Shot.to_frame()
    # 2.5 shots
    Prob25Shot = sims[:,:,shot_idx].copy()
    Prob25Shot = pd.DataFrame(Prob25Shot,columns=averages.ID).T
    Prob25Shot[Prob25Shot<2.5]=0
    Prob25Shot[Prob25Shot>=2.5]=1
    Prob25Shot=Prob25Shot.sum(axis=1)/10000
    Prob25Shot.name='Prob25Shot'
    Prob25Shot=Prob25Shot.to_frame()
    # 3 shots
    Prob3Shot = sims[:,:,shot_idx].copy()
    Prob3Shot = pd.DataFrame(Prob3Shot,columns=averages.ID).T
    Prob3Shot[Prob3Shot<3]=0
    Prob3Shot[Prob3Shot>=3]=1
    Prob3Shot=Prob3Shot.sum(axis=1)/10000
    Prob3Shot.name='Prob3Shot'
    Prob3Shot=Prob3Shot.to_frame()
    # 3.5 shots
    Prob35Shot = sims[:,:,shot_idx].copy()
    Prob35Shot = pd.DataFrame(Prob35Shot,columns=averages.ID).T
    Prob35Shot[Prob35Shot<3.5]=0
    Prob35Shot[Prob35Shot>=3.5]=1
    Prob35Shot=Prob35Shot.sum(axis=1)/10000
    Prob35Shot.name='Prob35Shot'
    Prob35Shot=Prob35Shot.to_frame()
    # 4 shots
    Prob4Shot = sims[:,:,shot_idx].copy()
    Prob4Shot = pd.DataFrame(Prob4Shot,columns=averages.ID).T
    Prob4Shot[Prob4Shot<4]=0
    Prob4Shot[Prob4Shot>=4]=1
    Prob4Shot=Prob4Shot.sum(axis=1)/10000
    Prob4Shot.name='Prob4Shot'
    Prob4Shot=Prob4Shot.to_frame()
    # 4.5 shots
    Prob45Shot = sims[:,:,shot_idx].copy()
    Prob45Shot = pd.DataFrame(Prob45Shot,columns=averages.ID).T
    Prob45Shot[Prob45Shot<4.5]=0
    Prob45Shot[Prob45Shot>=4.5]=1
    Prob45Shot=Prob45Shot.sum(axis=1)/10000
    Prob45Shot.name='Prob45Shot'
    Prob45Shot=Prob45Shot.to_frame()
    # Block Bonus
    BlockProb = sims[:,:,Block_idx].copy()
    BlockProb = pd.DataFrame(BlockProb,columns=averages.ID).T
    BlockProb[BlockProb<5]=0
    BlockProb[BlockProb>=5]=1
    BlockProb=BlockProb.sum(axis=1)/10000
    BlockProb.name='BlockProb'
    BlockProb=BlockProb.to_frame()
    # 1Block Props
    Prob1Block = sims[:,:,Block_idx].copy()
    Prob1Block = pd.DataFrame(Prob1Block,columns=averages.ID).T
    Prob1Block[Prob1Block<1]=0
    Prob1Block[Prob1Block>=1]=1
    Prob1Block=Prob1Block.sum(axis=1)/10000
    Prob1Block.name='Prob1Block'
    Prob1Block=Prob1Block.to_frame()
    # 1.5Block Props
    Prob15Block = sims[:,:,Block_idx].copy()
    Prob15Block = pd.DataFrame(Prob15Block,columns=averages.ID).T
    Prob15Block[Prob15Block<1]=0
    Prob15Block[Prob15Block>=1]=1
    Prob15Block=Prob15Block.sum(axis=1)/10000
    Prob15Block.name='Prob15Block'
    Prob15Block=Prob15Block.to_frame()
    # 2Block Props
    Prob2Block = sims[:,:,Block_idx].copy()
    Prob2Block = pd.DataFrame(Prob2Block,columns=averages.ID).T
    Prob2Block[Prob2Block<1]=0
    Prob2Block[Prob2Block>=1]=1
    Prob2Block=Prob2Block.sum(axis=1)/10000
    Prob2Block.name='Prob2Block'
    Prob2Block=Prob2Block.to_frame()
    # 2.5 Blocks
    Prob25Block = sims[:,:,Block_idx].copy()
    Prob25Block = pd.DataFrame(Prob25Block,columns=averages.ID).T
    Prob25Block[Prob25Block<1]=0
    Prob25Block[Prob25Block>=1]=1
    Prob25Block=Prob25Block.sum(axis=1)/10000
    Prob25Block.name='Prob25Block'
    Prob25Block=Prob25Block.to_frame()
    # Goal Prob
    ProbGoal = sims[:,:,goal_idx].copy()
    ProbGoal = pd.DataFrame(ProbGoal,columns=averages.ID).T
    ProbGoal[ProbGoal<0.5]=0
    ProbGoal[ProbGoal>=0.5]=1
    ProbGoal=ProbGoal.sum(axis=1)/10000
    ProbGoal.name='ProbGoal'
    ProbGoal=ProbGoal.to_frame()
    #    O/U 0.5 Assists
    Prob5Assist = sims[:,:,assist_idx].copy()
    Prob5Assist = pd.DataFrame(Prob5Assist,columns=averages.ID).T
    Prob5Assist[Prob5Assist<0.5]=0
    Prob5Assist[Prob5Assist>=0.5]=1
    Prob5Assist=Prob5Assist.sum(axis=1)/10000
    Prob5Assist.name='Prob5Assist'
    Prob5Assist=Prob5Assist.to_frame()
    #    O/U 1 Assists
    Prob1Assist = sims[:,:,assist_idx].copy()
    Prob1Assist = pd.DataFrame(Prob1Assist,columns=averages.ID).T
    Prob1Assist[Prob1Assist<1]=0
    Prob1Assist[Prob1Assist>=1.0]=1
    Prob1Assist=Prob1Assist.sum(axis=1)/10000
    Prob1Assist.name='Prob1Assist'
    Prob1Assist=Prob1Assist.to_frame()
    # O/U 1.5 Assists
    Prob15Assist = sims[:,:,assist_idx].copy()
    Prob15Assist = pd.DataFrame(Prob15Assist,columns=averages.ID).T
    Prob15Assist[Prob15Assist<1.5]=0
    Prob15Assist[Prob15Assist>=1.5]=1
    Prob15Assist=Prob15Assist.sum(axis=1)/10000
    Prob15Assist.name='Prob15Assist'
    Prob15Assist=Prob15Assist.to_frame()
    #%%
    skaters.set_index("ID", inplace=True)
    skaters = skaters.join(low["Floor"].round(1))
    skaters = skaters.join(high["Ceiling"].round(1))
    skaters = skaters.join(median[["Stochastic",'HDSC','shots','blocked','goals','assists']].round(1))
    skaters = skaters.join(ShotBonusProb)
    skaters = skaters.join(ProbHalfShot)
    skaters = skaters.join(Prob1Shot)
    skaters = skaters.join(Prob15Shot)
    skaters = skaters.join(Prob2Shot)
    skaters = skaters.join(Prob25Shot)
    skaters = skaters.join(Prob3Shot)
    skaters = skaters.join(Prob35Shot)
    skaters = skaters.join(Prob4Shot)
    skaters = skaters.join(Prob45Shot)

    skaters = skaters.join(BlockProb)
    skaters.reset_index(inplace=True)
    return skaters

def MLPrediction(
    skater_stats_df,
    goalie_stats_df,
    skater_games_df,
    goalie_games_df,
    position_type,
    line,
):
    print(position_type)
    # Determine player stats/opponent stats to be used based on position_type
    if position_type in ["Forward", "Defenseman"]:
        if "line" in NonFeatures:
            NonFeatures.remove("line")
        if 'season' not in NonFeatures:
            NonFeatures.append('season')
        game_proj_df = skater_games_df
        stats_df = skater_stats_df
        # Goalie Stats
        goalie_stats_df["line"] = 1
        goalie_stats_df.drop("line", axis=1, inplace=True)
        goalie_db_Features = [f"goalie_{c}" for c in OpposingGoalieColumns]
        # Opponent Stats
        opp_stats_df = stats_df.groupby(["opp", "game_date"]).sum()[OpposingTeamColumns]
        opp_stats_df = (
            opp_stats_df.groupby("opp")
            .apply(lambda x: rolling_average(x))
            .add_prefix("opp_")
        )

    else:
        NonFeatures.append("line")
        NonFeatures.remove('season')
        game_proj_df = goalie_games_df.groupby("full_name", as_index=False).first()
        stats_df = goalie_stats_df
        opp_stats_df = skater_stats_df

    game_proj_df.drop_duplicates(inplace=True)
    # Filter player down to active players for the week
    stats_df = stats_df[stats_df.player_id.isin(game_proj_df.player_id)]

    # Create prediction features frame by taking 15 game running average of player stats
    features = (
        stats_df.groupby(["player_id"])
        .apply(lambda x: rolling_average(x))
        .drop(NonFeatures, axis=1, errors="ignore")
    )

    # Reassign player_id,team to features frame
    features[["player_id", "team"]] = stats_df[["player_id", "team"]]
    if len(features) == 0:
        return

    # Get last row of features frame which is the latest running average for that player
    features = features.groupby(["player_id"], as_index=False).last()

    # Establish opponent/goalie features depending on position type being predicted
    if position_type in ["Forward", "Defenseman"]:
        goalie_features = goalie_stats_df.groupby("player_id")[
            goalie_db_Features
        ].apply(lambda x: rolling_average(x))
        goalie_features[["team", "game_date"]] = goalie_stats_df[["team", "game_date"]]
        goalie_features = goalie_features.groupby("team", as_index=False).last()
        opp_features = opp_stats_df.groupby("opp").last()
    else:
        opp_features = opp_stats_df.groupby(["opp", "game_date"]).sum()[
            OpposingTeamColumns
        ]
        opp_features["shooting_percentage"] = opp_features.goals / opp_features.shots
        opp_features["give_take_ratio"] = (
            opp_features.giveaways / opp_features.takeaways
        )
        opp_features = (
            opp_features.groupby("opp")
            .apply(lambda x: rolling_average(x))
            .add_prefix("opp_")
        )
        opp_features = opp_features.groupby("opp").last()
    # Merge Offense and Defense features
    features.set_index("player_id", inplace=True)
    features = features.join(
        game_proj_df[["player_id", "opp"]].set_index("player_id")
    ).reset_index()
    if position_type in ["Forward", "Defenseman"]:
        features = features.merge(goalie_features, on=["team"], how="left")
        features = features.merge(opp_features, on=["opp"], how="left")
    else:
        features = features.merge(opp_features, on=["opp"], how="left")

    # Setup projection frame
    ProjFrame = game_proj_df[NonFeatures]
    ProjFrame.set_index("player_id", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("player_id").drop(NonFeatures, axis=1, errors="ignore")
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame

    # Make projections
    for position in ProjFrame.position_type.unique():
        models = ModelDict[f"{position}{line}"]
        for method, model in models.items():
            ProjFrame.loc[ProjFrame.position_type == position, method] = model.predict(
                ProjFrame.loc[
                    ProjFrame.position_type == position,
                    features.drop(NonFeatures, axis=1, errors="ignore").columns,
                ]
            )
            ProjFrame.loc[
                (ProjFrame.position_type == position) & (ProjFrame[method] < 0), method
            ] = 0
    ProjFrame["ML"] = ProjFrame[models.keys()].mean(axis=1)
    # Predict Player Shares
    for position in ProjFrame.position_type.unique():
        if position == "Goalie":
            break
        for stat in ["goals", "assists", "shots", "blocked", "DKPts"]:
            model = SharesModelDict[f"{position}{line}_{stat}"]
            ProjFrame.loc[
                ProjFrame.position_type == position, f"proj_{stat}_share"
            ] = model.predict(
                ProjFrame.loc[
                    ProjFrame.position_type == position,
                    features.drop(NonFeatures, axis=1, errors="ignore").columns,
                ]
            )
            ProjFrame.loc[
                (ProjFrame.position_type == position)
                & (ProjFrame[f"proj_{stat}_share"] < 0),
                stat,
            ] = 0
    ProjFrame["ML"] = ProjFrame[models.keys()].mean(axis=1)
    ProjFrame.reset_index(inplace=True)
    return ProjFrame


#
def TeamStatsPredictions(team_games_df, team_stats_df):
    # Get 15 game running average of team statistics
    features = (
        team_stats_df.groupby(["team", "opp"])
        .apply(lambda x: rolling_average(x))
        .drop(NonFeatures, axis=1, errors="ignore")
    )
    features[["team", "opp"]] = team_stats_df[["team", "opp"]]
    if len(features) == 0:
        return

    # Get last row of team running average statistics
    features = features.groupby(["team"], as_index=False).last()

    # Get 15 game running average of oppponent statistics and latest average
    opp_features = (
        team_stats_df.groupby(["opp", "game_date"])
        .sum()
        .drop(TeamNonFeatures, axis=1, errors="ignore")
    )
    opp_features = (
        opp_features.groupby("opp")
        .apply(lambda x: rolling_average(x))
        .add_prefix("opp_")
    )
    opp_features = opp_features.groupby("opp").last()

    # Merge Offense and Defense features
    features = features.merge(opp_features, on=["opp"], how="left")
    if "proj_team_score" not in TeamNonFeatures:
        TeamNonFeatures.append("proj_team_score")

    # Setup projection
    ProjFrame = team_games_df[TeamNonFeatures]
    TeamNonFeatures.remove("proj_team_score")
    ProjFrame.set_index("team", inplace=True)
    ProjFrame = ProjFrame.join(
        features.set_index("team").drop(TeamNonFeatures, axis=1, errors="ignore")
    )
    ProjFrame.dropna(inplace=True)
    ProjFrame.drop_duplicates(inplace=True)
    if len(ProjFrame) == 0:
        return ProjFrame
    for stat in ["goals", "assists", "shots", "blocked", "DKPts"]:
        model = TeamModelDict[f"RF{stat}"]
        ProjFrame[f"proj_{stat}"] = model.predict(
            ProjFrame[features.drop(TeamNonFeatures, axis=1, errors="ignore").columns]
        )
        ProjFrame.loc[(ProjFrame[f"proj_{stat}"] < 0), f"proj_{stat}"] = 0
    ProjFrame.reset_index(inplace=True)
    ProjFrame["proj_opp_score"] = ProjFrame.team.apply(
        lambda x: ProjFrame[ProjFrame.opp == x].proj_team_score.values[0]
    )
    ProjFrame.loc[
        ProjFrame.proj_team_score < ProjFrame.proj_opp_score, "opp_Winner"
    ] = 1
    ProjFrame.opp_Winner.fillna(0, inplace=True)
    ProjFrame["opp_goalie_saves"] = ProjFrame.proj_shots - ProjFrame.proj_goals
    ProjFrame["opp_goalie_proj"] = (
        (ProjFrame.opp_goalie_saves * 0.7)
        - (ProjFrame.proj_goals * 3.5)
        + (ProjFrame.opp_Winner * 6)
    )
    ProjFrame.loc[ProjFrame.opp_goalie_saves >= 35, "opp_goalie_proj"] += 3
    return ProjFrame


def StochasticPrediction(
    player,
    position,
    Salary,
    line,
    moneyline,
    stats_df,
    opp,
    game_proj_df,
    position_type,
    contest_type,
):
    if position_type == "Goalie":
        STATS = DK_Goalie_Stats
        stats_df["line"] = 1
    else:
        STATS = DK_Skater_Stats
    if moneyline > 0:
        favorite = False
    else:
        favorite = True
    name = stats_df[stats_df.player_id == player].full_name.unique()
    n_games = 15
    feature_frame = pd.DataFrame({})
    opp_feature_frame = pd.DataFrame({})
    player_df = stats_df[stats_df.player_id == player][-n_games:]
    if contest_type == "Classic":
        opp_df = stats_df[
            (stats_df.opp == opp)
            & (stats_df.position == position)
            & ((stats_df.Salary >= Salary - 500) & (stats_df.Salary <= Salary + 500))
        ][-n_games:]
    else:
        opp_df = stats_df[
            (stats_df.opp == opp)
            & (stats_df.position == position)
            & (stats_df.line == line)
        ][-n_games:]
    if (len(player_df) == 0) | ((len(opp_df) == 0)):
        return pd.DataFrame(
            {"Floor": [np.nan], "Ceiling": [np.nan], "Stochastic": [np.nan]}
        )
    for stat in STATS:
        mean = player_df[stat].mean()
        std = player_df[stat].std()
        stats = np.random.normal(loc=mean, scale=std, size=10000)
        feature_frame[stat] = stats
        #
        opp_mean = opp_df[stat].mean()
        opp_std = opp_df[stat].std()
        opp_stats = np.random.normal(loc=opp_mean, scale=opp_std, size=10000)
        opp_feature_frame[stat] = opp_stats
    feature_frame.fillna(0, inplace=True)
    opp_feature_frame.fillna(0, inplace=True)
    feature_frame = feature_frame.mask(feature_frame.lt(0), 0)
    opp_feature_frame = opp_feature_frame.mask(opp_feature_frame.lt(0), 0)
    ProjectionsFrame = pd.DataFrame({})
    if player_df.DKPts.mean() >= opp_df.DKPts.mean():
        player_edge = True
    else:
        player_edge = False
    if (player_edge == True) & (favorite == True):
        player_weight = 1 - (1 / ((1 - (100 / (np.abs(moneyline) * -1)))))
    elif (player_edge == True) & (favorite == False):
        player_weight = 1 / ((1 - (100 / (np.abs(moneyline) * -1))))
    elif (player_edge == False) & (favorite == True):
        player_weight = 1 / ((1 - (100 / (np.abs(moneyline) * -1))))
    else:
        player_weight = 1 - (1 / ((1 - (100 / (np.abs(moneyline) * -1)))))
    opp_weight = 1 - player_weight
    if position_type == "Skater":
        Stochastic = getSkaterDKPts(feature_frame)
        opp_Stochastic = getSkaterDKPts(opp_feature_frame)
    else:
        Stochastic = getGoalieDKPts(feature_frame)
        opp_Stochastic = getGoalieDKPts(opp_feature_frame)
    feature_frame = pd.concat([feature_frame, opp_feature_frame]).reset_index(drop=True)
    ProjectionsFrame["Stochastic"] = (Stochastic * player_weight) + (
        opp_Stochastic * opp_weight
    )
    Floor = round(ProjectionsFrame.Stochastic.quantile(0.15), 1)
    if Floor < 0:
        Floor = 0
    if position_type == "Skater":
        Ceiling = round(ProjectionsFrame.Stochastic.quantile(0.85), 1)
        Stochastic = round(ProjectionsFrame.Stochastic.quantile(0.40), 1)
        ShotProb = len(feature_frame[feature_frame.shots >= 5]) / 20000
        BlockProb = len(feature_frame[feature_frame.blocked >= 3]) / 20000
        HDSC = feature_frame.HDSC.mean()
        opp_HDSC_allowed = opp_feature_frame.HDSC.mean()
        print(name, Stochastic)
        ProjFrame = pd.DataFrame(
            {
                "Floor": [Floor],
                "Ceiling": [Ceiling],
                "Stochastic": [Stochastic],
                "ShotProb": [ShotProb],
                "BlockProb": [BlockProb],
                "HDSC":[HDSC],
                "opp_HDSC":[opp_HDSC_allowed],
            }
        )
    else:
        Ceiling = round(ProjectionsFrame.Stochastic.quantile(0.85), 1)
        Stochastic = round(ProjectionsFrame.Stochastic.mean(), 1)
        print(name, Stochastic)
        ProjFrame = pd.DataFrame(
            {"Floor": [Floor], "Ceiling": [Ceiling], "Stochastic": [Stochastic]}
        )
    return ProjFrame


def TopDownPrediction(df):
    df["proj_goals"] = df[["proj_goals", "proj_team_score"]].mean(axis=1)
    df["TD_Proj"] = (
        (df.proj_shots_share * df.proj_shots * 1.5)
        + (df.proj_blocked_share * df.proj_blocked * 1.3)
        + (df.proj_goals_share * df.proj_goals * 8.5)
        + (df.proj_assists_share * df.proj_assists * 5)
    )
    df["TD_Proj"] += df.proj_DKPts_share * df.proj_DKPts
    df["TD_Proj"] /= 2
    return df
