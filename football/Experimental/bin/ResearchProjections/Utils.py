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

basedir = os.path.dirname(os.path.abspath(__file__))
datadir = f"{basedir}/../../data"
projdir = f"{datadir}/Projections"
etcdir = f"{basedir}/../../etc"

os.chdir(f"{basedir}/../")
from Models.ModelDict import *
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

def getOwnership(df):
    df = df[df.salary > 0]
    df["Value"] = (df.Projection / df.salary) * 1000
    OwnershipModelFeatures = [
        "total_line",
        "proj_team_score",
        "opp_Rank",
        "Projection",
        "Value",
    ]
    pos_frames = []
    zero_ownership=df[(df.position!='DST')&(df.Projection<5)]
    dst=df[df.position=='DST']
    ownership_df=df[df.Projection>5]
    df=pd.concat([dst,ownership_df])
    df.drop_duplicates(inplace=True)
    for pos in ["RB", "QB", "WR", "TE", "DST"]:
        pos_df = df[df.position == pos]
        print(pos)
        model = pickle.load(
            open(
                f"{etcdir}/model_pickles/{pos}_OwnershipModel.pkl",
                "rb",
                )
                )
        pos_df["OwnershipProj"] = model.predict(
            pos_df[OwnershipModelFeatures]
                )
        pos_df.loc[pos_df.OwnershipProj < 0, "OwnershipProj"] = np.random.choice(np.arange(0,1,0.01),size=len(pos_df[pos_df.OwnershipProj<0]))
        pos_df["OwnershipRank"] = pos_df.OwnershipProj.rank(
            ascending=False, method="min"
        )
        model=pickle.load(open(f'{etcdir}/model_pickles/{pos}_OwnershipRank_model.pkl','rb'))
        pos_df['ownership_proj_from_rank']=model.predict(pos_df['OwnershipRank'].values.reshape(-1,1))
        pos_df.loc[pos_df.ownership_proj_from_rank < 0, "ownership_proj_from_rank"] = np.random.choice(np.arange(0,1,0.01),size=len(pos_df[pos_df.ownership_proj_from_rank<0]))

        pos_df['Ownership']=pos_df[['OwnershipProj','ownership_proj_from_rank']].mean(axis=1)
        pos_frames.append(pos_df)
    pos_frames.append(zero_ownership)
    df = pd.concat(pos_frames)
    return df


def getImpliedOwnership(df):
    CRPs = {"QB": 99, "RB": 235.7, "WR": 344.1, "TE": 112.3, "DST": 105.1}
    for position in df.position.unique():
        cumUpside = df[df.position == position].sum().UpsideProb
        df.loc[df.position == position, "ImpliedOwnership"] = (
            df.loc[df.position == position, "UpsideProb"] / cumUpside
        )
        df.loc[df.position == position, "ImpliedOwnership"] = (
            df.loc[df.position == position, "ImpliedOwnership"] * CRPs[position]
        )
    df["Leverage"] = df.ImpliedOwnership / df.Ownership
    return df


def merge_dc_projections(df, dc):
    pos = df.position.values[0]
    if pos == "DST":
        df["DC_proj"] = np.nan
        return df
    depth_team = df.depth_team.values[0]
    if pos != "QB":
        if depth_team > 3:
            depth_team = 3
    else:
        if depth_team > 1:
            depth_team = 1
    column = f"proj_{pos}{depth_team}_DKPts"
    df = df.merge(dc[["team", column]], on="team", how="left")
    df.rename({column: "DC_proj"}, axis=1, inplace=True)
    return df


def getScaledProjection(df):
    scaled=df[~df[KnownFeatures].isna().any(axis=1)]
    scaled_nan=df[df[KnownFeatures].isna().any(axis=1)]
    for pos in scaled.position.unique():
        model_dict=ModelRanksDict[pos]
        for method in model_dict.keys():
            model=model_dict[method]
            scaled.loc[scaled.position==pos,f'ScaledProj_{method}']=model.predict(scaled.loc[scaled.position==pos,['ConsensusRank']+KnownFeatures])
        scaled['ScaledProj']=scaled[['ScaledProj_'+method for method in model_dict.keys()]].mean(axis=1)
    scaled.drop(['ScaledProj_'+method for method in model_dict.keys()],axis=1,inplace=True)
    scaled.loc[scaled.ScaledProj<0,'ScaledProj']=0
    df=pd.concat([scaled,scaled_nan])
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
    metrics = ["Projection", "TopDown", "Stochastic", "ML", "DC_proj", "PMM"]
    for metric in metrics:
        df[f"{metric}Rank"] = df.groupby(["position", "week", "season"])[
            metric
        ].apply(lambda x: x.rank(ascending=False, method="min"))
    df["Consensus"] = df[[metric + "Rank" for metric in metrics]].mean(axis=1)
    df["ConsensusRank"] = df.groupby(
        ["position", "week", "season"]
    ).Consensus.apply(lambda x: x.rank(method="min"))
    df["Rank"] = df.groupby(["position", "week", "season"]).DKPts.apply(
        lambda x: x.rank(ascending=False, method="min")
    )
    # df.drop(['Consensus']+[metric+'Rank' for metric in metrics],axis=1,inplace=True)
    df = getScaledProjection(df)
    return df
