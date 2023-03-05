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


def processRankings(proj,Scaled):
    # Rank players by different metrics
    for method in ['Stochastic','ML','PMM','TD_Proj','DDProb','TDProb','TD_Proj2']:
        proj.loc[proj.RG_projection==0,method]=0
    proj["MyRank"] = proj.Stochastic.rank(ascending=False, method="min")
    proj["RGRank"] = proj.RG_projection.rank(ascending=False, method="min")
    proj["MLRank"] = proj.ML.rank(ascending=False, method="min")
    proj["PMMRank"] = proj.PMM.rank(ascending=False, method="min")
    proj["TDRank"] = proj.TD_Proj.rank(ascending=False, method="min")
    proj["TDRank2"] = proj.TD_Proj2.rank(ascending=False, method="min")
    proj["DDRank"] = proj.DDProb.rank(ascending=False)
    proj["TrDRank"] = proj.TDProb.rank(ascending=False)
    proj['ConsRank']=proj[['MyRank','RGRank','MLRank','PMMRank','TDRank','TDRank2','DDRank','TrDRank']].mean(axis=1).rank()
    for method in ["RF", "GB"]:
        model = pickle.load(
                open(
                    f"{etcdir}/model_pickles/{method}_dkpts_Rank_model.pkl",
                    "rb",
                )
            )
        proj = proj[proj.ConsRank.isna() == False]
        proj[f'{method}_Rank'] = model.predict(proj.ConsRank[:, np.newaxis])
    proj["ScaledProj"] = proj[["RF_Rank", "GB_Rank"]].mean(axis=1)
    proj.loc[proj.RG_projection==0,'ScaledProj']=0
    if Scaled==True:
        proj["Projection"] = proj[
            ["Stochastic", "RG_projection", "ML", "TD_Proj","TD_Proj2", "ScaledProj"]
        ].mean(axis=1)
    else:
        proj["Projection"] = proj[
            ["Stochastic", "RG_projection", "ML", "TD_Proj","TD_Proj2"]
        ].mean(axis=1)
    return proj
