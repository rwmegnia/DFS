#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:03:15 2022

@author: robertmegnia

Function to get ownership projections
"""
import os
import pandas as pd
import numpy as np
import pickle
from RosterUtils import reformatName 
import nfl_data_py as nfl
basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
etcdir = f"{basedir}/../../etc"
os.chdir("../")
from Models.ModelDict import ModelRanksDict

os.chdir("./RealTimeProjections")
#%%
KnownFeatures = [
    "total_line",
    "proj_team_score",
    "spread_line",
    "opp_Rank",
    "Adj_opp_Rank",
    "salary",
    "depth_team",
]

def retrieveVariance(player,db,min_deviation=False):
    db=db[db.full_name==player]
    if len(db)<5:
        return 0
    db['variance']=db.apply(lambda x: (x['Projection']-x.DKPts)/x['Projection'],axis=1)
    if min_deviation==True:
        return db.variance.quantile(0.10)
    else:
        return db.variance.quantile(0.90)
    
def getPlayerVariance(df,week):
    proj_db=pd.read_csv(f'{datadir}/Projections/2022/megnia_projections_verified.csv')
    proj_db=proj_db[proj_db.week<week]
    proj_db=proj_db[proj_db['Projection']>0]
    df['min_deviation']=proj_db.full_name.apply(lambda x: retrieveVariance(x,proj_db,True))
    df['max_deviation']=proj_db.full_name.apply(lambda x: retrieveVariance(x,proj_db,False))
    max_deviation = df.max_deviation.quantile(0.95)
    min_deviation = df[df.min_deviation>0].min_deviation.quantile(0.05)
    df.loc[df.min_deviation<min_deviation,'min_deviation']=min_deviation
    df.loc[df.min_deviation<0,'min_deviation']=0
    df.loc[df.max_deviation>max_deviation,'max_deviation']=max_deviation
    df.loc[df.min_deviation.isna()==True,'min_deviation']=min_deviation
    df.loc[df.max_deviation.isna()==True,'max_deviation']=max_deviation
    return df

def getOwnership(df):
    # NonMain = df[df.Slate == "Full"]
    # df = df[(df.salary > 0) & (df.Slate == "Main")]
    df=df[df.salary>0]
    df["Value"] = (df.Projection / df.salary) * 1000
    OwnershipModelFeatures = [
        "total_line",
        "proj_team_score",
        "opp_Rank",
        "Projection",
        "Value",
    ]
    df.loc[df.ownership_proj==0,'ownership_proj']=np.nan
    pos_frames = []
    for pos in ["RB", "QB", "WR", "TE", "DST"]:
        print(pos)
        pos_df = df[df.position == pos]
        model = pickle.load(
            open(
                f"{etcdir}/model_pickles/{pos}_EN_RosterPercent_model.pkl",
                "rb",
            )
        )
        pos_df["Ownership"] = model.predict(pos_df[OwnershipModelFeatures])
        pos_df.loc[pos_df.Ownership < 0, "Ownership"] = np.random.choice(
            np.arange(0.01, 1, 0.3)
        )
        # pos_df.loc[
        #     (pos_df.ownership_proj.isna() == True) & (pos_df.position != "DST"),
        #     "Ownership",
        # ] = np.nan
        pos_df["AvgOwnership"] = pos_df[["Ownership", "ownership_proj"]].mean(
            axis=1
        )
        pos_df["OwnershipRank"] = pos_df.AvgOwnership.rank(
            ascending=False, method="min"
        )
        print(pos)
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_OwnershipRank_model.pkl", "rb")
        )
        pos_df.loc[
            pos_df.OwnershipRank.isna() == False, "ownership_proj_from_rank"
        ] = model.predict(
            pos_df.loc[
                pos_df.OwnershipRank.isna() == False, "OwnershipRank"
            ].values.reshape(-1, 1)
        )
        pos_df.loc[
            pos_df.ownership_proj_from_rank < 0, "ownership_proj_from_rank"
        ] = np.random.choice(
            np.arange(0, 1, 0.01),
            size=len(pos_df[pos_df.ownership_proj_from_rank < 0]),
        )
        pos_df["AvgOwnership"] = pos_df[
            ["ownership_proj", "ownership_proj_from_rank"]
        ].mean(axis=1)
        pos_frames.append(pos_df)
    df = pd.concat(pos_frames)
    #df = pd.concat([df, NonMain])
    df.AvgOwnership.fillna(0, inplace=True)
    return df


def getImpliedOwnership(df):
    CRPs = {"QB": 99, "RB": 235.7, "WR": 344.1, "TE": 112.3, "DST": 105.1}
    for position in df.position.unique():
        if position == "K":
            continue
        cumUpside = df[df.position == position].sum().UpsideProb
        df.loc[df.position == position, "ImpliedOwnership"] = (
            df.loc[df.position == position, "UpsideProb"] / cumUpside
        )
        df.loc[df.position == position, "ImpliedOwnership"] = (
            df.loc[df.position == position, "ImpliedOwnership"] * CRPs[position]
        )
    df["Leverage"] = df.ImpliedOwnership / df.AvgOwnership
    return df


def merge_dc_projections(df, dc):
    pos = df.position.values[0]
    if pos == "DST":
        df["DC_proj"] = np.nan
        return df
    depth_team = df.depth_team.values[0]
    if pos in ["RB", "TE"]:
        if depth_team > 3:
            df["DC_proj"] = 0
            return df
    elif pos == "WR":
        if depth_team > 5:
            df["DC_proj"] = 0
            return df
    else:
        if depth_team > 1:
            df["DC_proj"] = 0
            return df
    column = f"proj_{pos}{int(depth_team)}_DKPts"
    df = df.merge(dc[["team", column]], on="team", how="left")
    df.rename({column: "DC_proj"}, axis=1, inplace=True)
    return df


def getScaledProjection(df):
    scaled = df[~df[KnownFeatures].isna().any(axis=1)]
    scaled_nan = df[df[KnownFeatures].isna().any(axis=1)]
    for pos in scaled.position.unique():
        model_dict = ModelRanksDict[pos]
        for method in model_dict.keys():
            model = model_dict[method]
            scaled.loc[
                scaled.position == pos, f"ScaledProj_{method}"
            ] = model.predict(
                scaled.loc[
                    scaled.position == pos, ["ConsensusRank"] + KnownFeatures
                ]
            )
        scaled["ScaledProj"] = scaled[
            ["ScaledProj_" + method for method in model_dict.keys()]
        ].mean(axis=1)
    scaled.drop(
        ["ScaledProj_" + method for method in model_dict.keys()],
        axis=1,
        inplace=True,
    )
    scaled.loc[scaled.ScaledProj < 0, "ScaledProj"] = 0
    df = pd.concat([scaled, scaled_nan])
    return df


def getConsensusRanking(df):
    """
    Rank players with each of the follow metrics and take the average
    to get a consensus ranking

    Projection
    TopDown
    Stochastic
    ML
    DC_proj
    PMMRank
    UpsideProb
    LeverageScore

    """
    metrics = [
        "Projection",
        "TopDown",
        "Stochastic",
        "DC_proj",
        "PMM",
        "RG_projection",
        "FP_Proj",
        "FP_Proj2",
        "FP_Proj_Full",
    ]
    for metric in metrics:
        df[f"{metric}Rank"] = df.groupby(["position", "week", "season"])[
            metric
        ].apply(lambda x: x.rank(ascending=False, method="min"))
    df["Consensus"] = df[[metric + "Rank" for metric in metrics]].mean(axis=1)
    df["ConsensusRank"] = df.groupby(
        ["position", "week", "season"]
    ).Consensus.apply(lambda x: x.rank(method="min"))
    df.drop(
        ["Consensus"] + [metric + "Rank" for metric in metrics],
        axis=1,
        inplace=True,
    )
    df = getScaledProjection(df)
    return df


def BiasCorrection(df):
    pos_frames = []
    for pos in df.position.unique():
        print(pos)
        pos_frame = df[df.position == pos]
        bias_df = pd.read_csv(
            f"{datadir}/BiasCorrections/{pos}BiasCorrections.csv"
        )
        pos_frame["Projection"] = pos_frame.apply(
            lambda x: x.Projection
            + bias_df[bias_df.ConsensusRank == x.ConsensusRank].bias.values[0]
            if x.ConsensusRank in bias_df.ConsensusRank
            else x.Projection,
            axis=1,
        )
        pos_frames.append(pos_frame)
    df = pd.concat(pos_frames)
    return df


def reformatFantasyPros(df):
    df.rename(
        {
            "Position": "position",
            "Name": "full_name",
            "Team": "team",
            "Opponent": "opp",
            "Proj Pts.": "FP_Proj",
            "Salary": "salary",
        },
        axis=1,
        inplace=True,
    )
    df.team.replace({"JAC": "JAX"}, inplace=True)
    df.team.replace({"LAR": "LA"}, inplace=True)
    df.opp.replace({"JAC": "JAX"}, inplace=True)
    df.opp.replace({"LAR": "LA"}, inplace=True)
    df.full_name.replace({'Gabe Davis':'Gabriel Davis',
                          'Dee Eskridge':'Dwayne Eskridge',
                          'Jody Forston':'Joe Fortson',
                          'Kenneth Walker III':'Kenneth Walker'},inplace=True)
    df = reformatName(df)
    df.loc[df.position == "DST", "RotoName"] = df.loc[
        df.position == "DST", "team"
    ]
    df.loc[df.position == "DST", "full_name"] = df.loc[
        df.position == "DST", "team"
    ]
    return df[
        [
            "position",
            "full_name",
            "salary",
            "team",
            "opp",
            "FP_Proj",
            "RotoName",
        ]
    ]


def reformatGI_Aggregated(df):
    df=df[df.Player.isna()==False]
    df.rename(
        {
            "DK Position": "position",
            "Player": "full_name",
            "Team": "team",
            "Opponent": "opp",
            "mean_proj": "GI_agg_proj",
        },
        axis=1,
        inplace=True,
    )
    df.loc[df.position=='DST','full_name']=df.loc[df.position=='DST','team']
    df = reformatName(df)
    df.loc[df.position == "DST", "RotoName"] = df.loc[
        df.position == "DST", "team"
    ]
    df.loc[df.position == "DST", "full_name"] = df.loc[
        df.position == "DST", "team"
    ]
    return df[
        [
            "position",
            "full_name",
            "team",
            "opp",
            "GI_agg_proj",
            "RotoName",
        ]
    ]

def getDKPts(df, stat_type):
    if stat_type == "Offense":
        df.loc[df.pass_yards >= 300, "pass_yards"] = (
            df.loc[df.pass_yards >= 300, "pass_yards"] + 30
        )
        df.loc[df.rush_yards >= 100, "rush_yards"] = (
            df.loc[df.rush_yards >= 100, "rush_yards"] + 30
        )
        df.loc[df.rec_yards >= 100, "rec_yards"] = (
            df.loc[df.rec_yards >= 100, "rec_yards"] + 30
        )
        return (
            (df.pass_yards * 0.04)
            + (df.rush_yards * 0.1)
            + (df.pass_td * 4)
            + (df.rush_td * 6)
            + df.rec
            + (df.rec_yards * 0.1)
            + (df.rec_td * 6)
            - (df.int)
            - (df.fumbles_lost)
        )
    else:
        points = (
            (df.fumble_recoveries * 2)
            + (df.interception * 2)
            + df.sack
            + (df.blocks * 6)
            + (df.safety * 2)
            + (df.return_touchdown * 6)
        )
        points_allowed = df.points_allowed.values[0]
        if points_allowed == 0:
            return points + 10
        elif (points_allowed > 0) & (points_allowed <= 6):
            return points + 7
        elif (points_allowed > 6) & (points_allowed <= 13):
            return points + 4
        elif (points_allowed > 13) & (points_allowed <= 20):
            return points + 1
        elif (points_allowed > 20) & (points_allowed <= 27):
            return points
        elif (points_allowed > 27) & (points_allowed <= 34):
            return points - 1
        else:
            return points - 4


def getWeightedProjections(projections,GI,Nick):
    # Weighted Projection
    weighted_projections=projections[(projections.TopDown.isna()==False)&
                                     (projections.PMM.isna()==False)&
                                     (projections.Stochastic.isna()==False)&
                                     (projections.DC_proj.isna()==False)&
                                     (projections.ScaledProj.isna()==False)]
    weighted_projections.loc[weighted_projections.position != "DST", "wProjection"] = (
        (weighted_projections.loc[weighted_projections.position != "DST", "TopDown"] * 0.25)
        + (weighted_projections.loc[weighted_projections.position != "DST", "PMM"] * 0.40)
        + (weighted_projections.loc[weighted_projections.position != "DST", "Stochastic"] * 0.20)
        + (weighted_projections.loc[weighted_projections.position != "DST", "DC_proj"] * 0.10)
        + (weighted_projections.loc[weighted_projections.position != "DST", "ScaledProj"] * 0.05)
    )
    projections=projections[~projections.gsis_id.isin(weighted_projections.gsis_id)]
    projections=pd.concat([projections,weighted_projections])
    projections.loc[
        projections.position == "DST", "wProjection"
    ] = projections.loc[projections.position == "DST", "Projection"]
    projections.loc[
        projections.wProjection.isna()==True,'wProjection'
        ] = projections.loc[projections.wProjection.isna()==True,"Projection"]
    if (GI==True)&(Nick==True):
        projections["wProjection"] = projections[
            ["wProjection", 
             "FP_Proj",
             "FP_Proj2", 
             "RG_projection", 
             "FP_Proj_Full",
             "NicksAgg",
             "GI"]
        ].mean(axis=1)
    elif (GI==True)&(Nick==False):
        projections["wProjection"] = projections[
            ["wProjection", 
             "FP_Proj",
             "FP_Proj2", 
             "RG_projection", 
             "FP_Proj_Full",
             "GI"]
        ].mean(axis=1)
    elif (GI==False)&(Nick==True):
        projections["wProjection"] = projections[
            ["wProjection", 
             "FP_Proj",
             "FP_Proj2", 
             "RG_projection", 
             "FP_Proj_Full",
             "NicksAgg"]
        ].mean(axis=1)
    else:
        projections["wProjection"] = projections[
            ["wProjection", 
             "FP_Proj",
             "FP_Proj2", 
             "RG_projection", 
             "FP_Proj_Full",]
        ].mean(axis=1)
    projections.loc[
        projections.wProjection.isna() == True, "wProjection"
    ] = projections.loc[projections.wProjection.isna() == True, "Projection"]
    return projections

def reformatGridIron(df,ids):
    df=df.merge(ids,on=['player_id'],how='left')
    df.rename({'passing_yards':'pass_yards',
               'rushing_yards':'rush_yards',
               'passing_touchdowns':'pass_td',
               'rushing_touchdowns':'rush_td',
               'receiving_receptions':'rec',
               'receiving_yards':'rec_yards',
               'receiving_touchdowns':'rec_td',
               'passing_interceptions':'int',
               'offense_fumble_lost':'fumbles_lost',
               'name':'full_name',
               'position_id':'position'},
              axis=1,inplace=True)
    df['GI']=getDKPts(df,'Offense')
    df=reformatName(df)
    return df

def reformatNicksAgg(df):
    df.rename({'Player':'full_name',
               'Points':'NicksAgg'},
              axis=1,inplace=True)
    df.full_name.replace({'Gabe Davis':'Gabriel Davis'},inplace=True)
    df.loc[df.position=='DST','full_name']=df.loc[df.position=='DST','team']
    df=reformatName(df)
    df.drop(['index',
             'DKSlateID',
             'full_name',
             'merge_name',
             'last_updated_at_et'],axis=1,inplace=True,errors='ignore')
    df['high_proj']=df[['espn_proj',
                        'line_star_proj',
                        'stokastic_proj',
                        'theblitz_proj',
                        'rotogrinders_proj',
                        'sabersim_proj',
                        'etr_proj',
                        'paulsen_proj',
                        'ows_proj',
                        'rotowire_proj',]].max(axis=1)
    df['low_proj']=df[['espn_proj',
                        'line_star_proj',
                        'stokastic_proj',
                        'theblitz_proj',
                        'rotogrinders_proj',
                        'sabersim_proj',
                        'etr_proj',
                        'paulsen_proj',
                        'ows_proj',
                        'rotowire_proj',]].min(axis=1)
    df['std_proj']=df[['espn_proj',
                        'line_star_proj',
                        'stokastic_proj',
                        'theblitz_proj',
                        'rotogrinders_proj',
                        'sabersim_proj',
                        'etr_proj',
                        'paulsen_proj',
                        'ows_proj',
                        'rotowire_proj',]].std(axis=1)
    return df

def mergeStats(season,week,proj,off_db,Nick=False):
    proj=proj.groupby('gsis_id',as_index=False).first()
    if Nick==True:
        proj=proj[[
                     'full_name',
                     'gsis_id',
                     'team',
                     'opp',
                     'position',
                     'salary',
                     'total_line',
                     'proj_team_score',
                     'spread_line',
                     'proj_team_fpts',
                     'proj_team_rushing_fpts',
                     'proj_team_receiving_fpts',
                     'proj_team_passing_fpts',
                     'depth_team',
                     'injury_designation',
                     'game_location',
                     'game_day',
                     'UpsideScore',
                     'RotoName',
                     'game_time',
                     'season',
                     'game_date',
                     'Slate',
                     'week',
                     'ID',
                     'wProjection',
                     'NicksAgg',
                     'Floor',
                     'Ceiling',
                     'AvgOwnership',
                     'own']]
    else:
        proj=proj[[
                     'full_name',
                     'gsis_id',
                     'team',
                     'opp',
                     'position',
                     'salary',
                     'total_line',
                     'proj_team_score',
                     'spread_line',
                     'proj_team_fpts',
                     'proj_team_rushing_fpts',
                     'proj_team_receiving_fpts',
                     'proj_team_passing_fpts',
                     'depth_team',
                     'injury_designation',
                     'game_location',
                     'game_day',
                     'UpsideScore',
                     'RotoName',
                     'game_time',
                     'season',
                     'game_date',
                     'Slate',
                     'week',
                     'ID',
                     'wProjection',
                     'Floor',
                     'Ceiling',
                     'AvgOwnership']]
    off_db=pd.read_csv(f'{datadir}/game_logs/{season}/{season}_Offense_GameLogs.csv')
    off_db=reformatName(off_db)
    off_db.drop(off_db[(off_db.position=='QB')&(off_db.depth_team!=1)].index,inplace=True)
    stats=off_db.groupby(['gsis_id']).agg({
        'pass_yds_per_att':np.mean,
        'passer_rating':np.mean,
        'pass_air_yards':np.mean,
        'DKPts':np.mean,
        'rush_yds_per_att':np.mean,
        'rush_value':np.mean,
        'Usage':np.mean,
        'rush_redzone_looks':np.mean,
        'targets':np.mean,
        'target_share':np.mean,
        'air_yards_share':np.mean,
        'target_value':np.mean,
        'wopr':np.mean,
        'exDKPts':np.mean,
        'poe':np.mean,
        'adot':np.mean,
        'yac':np.mean,
        'rec_redzone_looks':np.mean,
        'offensive_snapcounts':np.mean,
        'HVU':np.mean,
        'offensive_snapcount_percentage':np.mean,
        })
    stats=stats.reset_index().set_index(['gsis_id'])
    proj.set_index('gsis_id',inplace=True)
    proj=proj.join(stats)
    proj.loc[proj.position=='DST','team']=proj.loc[proj.position=='DST'].index
    proj['team']=proj['team'].apply(lambda x: x.upper())
    def_qb_stats=off_db[(off_db.position=='QB')&(off_db.depth_team==1)].groupby(['opp']).agg({
        'pass_yds_per_att':np.mean,
        'passer_rating':np.mean,
        'pass_air_yards':np.mean,
        'pass_td':np.sum,
        })
    def_qb_stats=def_qb_stats.add_prefix('opp_')
    def_qb_stats=def_qb_stats.add_suffix('_allowed')
    def_qb_stats_rank = def_qb_stats.rank().add_suffix('_rank')
    def_rb_stats=off_db[(off_db.position=='RB')&(off_db.depth_team<3)].groupby(['opp']).agg({
        'rush_yds_per_att':np.mean,
        'rush_td':np.sum,
        })
    def_rb_stats=def_rb_stats.add_prefix('opp_')    
    def_rb_stats=def_rb_stats.add_suffix('_allowed')
    def_rb_stats_rank=def_rb_stats.rank().add_suffix('_rank')
    def_wr_stats=off_db[(off_db.position=='WR')&(off_db.depth_team<4)].groupby(['opp']).agg({
        'adot':np.mean,
        'yac':np.mean,
        'rec_air_yards':np.mean,
        'rec_td':np.sum
        })
    def_wr_stats=def_wr_stats.add_prefix('opp_')
    def_wr_stats=def_wr_stats.add_suffix('_allowed')
    def_wr_stats_rank=def_wr_stats.rank().add_suffix('_rank')
    def_te_stats=off_db[(off_db.position=='TE')&(off_db.depth_team<=2)].groupby(['opp']).agg({
        'adot':np.mean,
        'yac':np.mean,
        'rec_air_yards':np.mean,
        'rec_td':np.sum
        })
    def_te_stats=def_te_stats.add_prefix('opp_')
    def_te_stats=def_te_stats.add_suffix('_allowed_te')
    def_te_stats_rank=def_te_stats.rank().add_suffix('_rank')
    proj.reset_index(inplace=True)
    proj=proj.merge(def_qb_stats,on='opp',how='left')
    proj=proj.merge(def_rb_stats,on='opp',how='left')
    proj=proj.merge(def_wr_stats,on='opp',how='left')
    proj=proj.merge(def_te_stats,on='opp',how='left')
    proj=proj.merge(def_qb_stats_rank,on='opp',how='left')
    proj=proj.merge(def_rb_stats_rank,on='opp',how='left')
    proj=proj.merge(def_wr_stats_rank,on='opp',how='left')
    proj=proj.merge(def_te_stats_rank,on='opp',how='left')
    # Points Allowed
    def_avg=off_db.groupby(['opp','position'],as_index=False).mean()
    def_avg['OppRank']=def_avg.groupby('position').DKPts.rank()   
    proj=proj.merge(def_avg[['opp','position','OppRank']],on=['opp','position'],how='left')
    #
    headshots=pd.read_csv(f'{datadir}/Projections/{season}/megnia_projections.csv')
    headshots=headshots.groupby(['gsis_id','week'],as_index=False).first()
    headshots=reformatName(headshots)
    headshots=headshots[headshots.week==week]
    proj.rename({'GI':'GridIron',
                      'Projection':'RobsProjection',
                      'proj_team_score':'ImpliedTeamTotal',
                      'total_line':'Over/Under',
                      'spread_line':'spread'},axis=1,inplace=True)
    proj.loc[proj.position=='DST','RotoName']=proj.loc[proj.position=='DST','RotoName'].apply(lambda x: x.upper())
    headshots=headshots.groupby('RotoName').first()['headshot_url'].to_frame()
    proj=proj.merge(headshots,on='RotoName',how='left')
    proj['week']=week
    proj['season']=season
    teams=nfl.import_team_desc()
    teams.replace({'LAR':'LA'},inplace=True)
    teams.rename({'team_abbr':'team'},axis=1,inplace=True)
    dst=proj[proj.position=='DST']
    dst.drop('headshot_url',axis=1,inplace=True)
    dst['team']=dst['RotoName']
    dst=dst.merge(teams[['team','team_logo_espn']],on='team',how='left')
    league_logo=teams.team_league_logo.unique()[0]
    dst.rename({'team_logo_espn':'headshot_url'},axis=1,inplace=True)
    dst=dst[proj.columns]
    proj=pd.concat([proj[proj.position!='DST'],dst])
    proj.loc[proj.headshot_url.isnull(),'headshot_url']=league_logo
    proj=proj.merge(teams[['team','team_logo_espn','team_wordmark']],on='team',how='left')
    proj.rename({'team_logo_espn':'team_logo',
                    'team_wordmark':'team_logo_wordmark'},axis=1,inplace=True)
    teams.rename({'team':'opp'},axis=1,inplace=True)
    proj=proj.merge(teams[['opp','team_logo_espn','team_wordmark']],on='opp',how='left')
    proj.rename({'team_logo_espn':'opp_logo',
                    'team_wordmark':'opp_wordmark'},axis=1,inplace=True)

    proj.drop_duplicates(inplace=True)
    return proj
