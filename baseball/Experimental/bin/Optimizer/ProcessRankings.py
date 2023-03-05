#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:33:41 2022

@author: robertmegnia
"""
import pickle
import os
from datetime import datetime
import numpy as np
import pandas as pd
from pydfs_lineup_optimizer.tz import get_timezone
from pytz import timezone

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc"
datadir = f"{basedir}/../../data"


def processRankings(proj, game_date, Scaled=True):
    # Rank batters by different metrics
    pos_frames=[]
    for pos in ['Batter','Pitcher']:
        if pos=='Pitcher':
            pos_frame=proj[proj.RotoPosition.isin(['P','SP'])]
            pos_frame["MyRank"] = pos_frame.Stochastic.rank(ascending=False, method="min")
            pos_frame["RGRank"] = pos_frame.RG_projection.rank(ascending=False, method="min")
            pos_frame["KRank"] = pos_frame.K_prcnt.rank(ascending=False, method="min")
            pos_frame["ERARank"]=pos_frame.era.rank( method="min")
            pos_frame["WHIPRank"]=pos_frame.WHIP.rank( method="min")
            pos_frame["MLRank"] = pos_frame.ML.rank(ascending=False, method="min")
            pos_frame["PMMRank"] = pos_frame.PMM.rank(ascending=False, method="min")
            pos_frame["ConsRank"] = pos_frame[
                ["MyRank", "RGRank", "MLRank", "KRank","PMMRank","ERARank","WHIPRank"]
            ].mean(axis=1)
        else:
            pos_frame=proj[~proj.RotoPosition.isin(['P','SP'])]
            pos_frame["MyRank"] = pos_frame.Stochastic.rank(ascending=False, method="min")
            pos_frame["RGRank"] = pos_frame.RG_projection.rank(ascending=False, method="min")
            pos_frame["ISORank"] = pos_frame.ISO.rank(ascending=False, method="max")
            pos_frame["MLRank"] = pos_frame.ML.rank(ascending=False, method="min")
            pos_frame["PMMRank"] = pos_frame.ML.rank(ascending=False, method="min")
            pos_frame['wOBARank']=pos_frame.wOBA.rank(ascending=False,method='max')
            pos_frame['exWOBARank']=pos_frame.exWOBA.rank(ascending=False,method='max')
            pos_frame["ConsRank"] = pos_frame[
                ["MyRank", "ISORank", "RGRank", "MLRank","PMMRank","wOBARank",'exWOBARank']
            ].mean(axis=1)
        for method in ["RF", "GB"]:
            model = pickle.load(
                open(
                    f"{etcdir}/model_pickles/{pos}_{method}_DKPts_Rank_model.pkl",
                    "rb",
                )
            )
            pos_frame = pos_frame[pos_frame.ConsRank.isna() == False]
            pos_frame[f'{method}_Rank'] = model.predict(pos_frame.ConsRank[:, np.newaxis])
        pos_frame["ScaledProj"] = pos_frame[["RF_Rank", "GB_Rank"]].mean(axis=1)
        pos_frames.append(pos_frame)
    proj = pd.concat(pos_frames)
    if Scaled==True:
        proj["Projection"] = proj[
            ["Stochastic", "RG_projection","ScaledProj","PMM"]
        ].mean(axis=1)
    else:
        proj["Projection"] = proj[
            ["Stochastic",'ML', "RG_projection","PMM"]
        ].mean(axis=1)
    try:
        proj["game_date"] = proj["game_date"].apply(
            lambda x: datetime.strptime(x, "%Y-%m-%d").replace(
                tzinfo=timezone(get_timezone())
            )
        )
    except ValueError:
        proj.game_date = pd.to_datetime(proj.game_date)
    # proj.fillna(0, inplace=True)
    return proj
