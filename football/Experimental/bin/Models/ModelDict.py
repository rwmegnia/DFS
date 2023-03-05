#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:22:59 2021

@author: robertmegnia

Builds dictionaries for various machine learning models that are
kept in the etc/model_pickles directory
"""
import pickle
import os

basedir = os.path.dirname(os.path.abspath(__file__))
etcdir = f"{basedir}/../../etc/"

# Load Models for Player DKPts Projections
ModelDict = {"QB": {}, "RB": {}, "WR": {}, "TE": {}, "DST": {}}
for pos in ModelDict.keys():
    for method in ["BR", "EN", "NN", "RF", "GB", "Tweedie"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_{method}_DKPts_model.pkl", "rb")
        )
        ModelDict[pos][method] = model

# Load Models for Rookie Player DKPts Projections

RookieModelDict = {
    "QB": {},
    "RB": {},
    "WR": {},
    "TE": {},
}
for pos in RookieModelDict.keys():
    for method in ["BR", "EN", "NN", "RF", "GB", "Tweedie"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_{method}_Rookie_model.pkl", "rb")
        )
        RookieModelDict[pos][method] = model

# Load Models for DepthChart Projections
TeamModelDict = {
    "QB1_DKPts": {},
    "RB1_DKPts": {},
    "RB2_DKPts": {},
    "RB3_DKPts": {},
    "WR1_DKPts": {},
    "WR2_DKPts": {},
    "WR3_DKPts": {},
    "WR4_DKPts": {},
    "WR5_DKPts": {},
    "TE1_DKPts": {},
    "TE2_DKPts": {},
    "TE3_DKPts": {},
}

for pos in TeamModelDict.keys():
    for method in ["BR", "EN", "NN", "RF", "GB", "Tweedie"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_{method}_model.pkl", "rb")
        )
        TeamModelDict[pos][method] = model

## Load Models for Team Stat Projetions
TeamStatsModelDict = {
    "offense_snaps":{},
    "pass_attempt": {},
    "rush_attempt": {},
    "team_fpts": {},
    "passing_fpts": {},
    "rushing_fpts": {},
    "receiving_fpts": {},
}

for stat in TeamStatsModelDict.keys():
    for method in ["BR", "EN", "NN", "RF", "GB", "Tweedie"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{stat}_{method}_model.pkl", "rb")
        )
        TeamStatsModelDict[stat][method] = model

## Load Models for Points Share Projections
ShareStatsModelDict = {}
for pos in ["QB", "RB", "WR", "TE"]:
    ShareStatsModelDict[pos] = {}
    for stat in ["rushing_DKPts_share", "receiving_DKPts_share"]:
        if (pos == "QB") & (stat == "receiving_DKPts_share"):
            continue
        ShareStatsModelDict[pos][stat] = {}
        for method in ["BR", "EN", "NN", "RF", "GB", "Tweedie"]:
            model = pickle.load(
                open(f"{etcdir}/model_pickles/{pos}_{method}_{stat}_model.pkl", "rb")
            )
            ShareStatsModelDict[pos][stat][method] = model

## Load Models for Points derived from Rank
# Load Models for Player DKPts Projections
ModelRanksDict = {"QB": {}, "RB": {}, "WR": {}, "TE": {}, "DST": {}}
for pos in ModelRanksDict.keys():
    for method in ["BR", "EN", "NN", "RF", "GB", "Tweedie"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_{method}_DKPts_Rank_model.pkl", "rb")
        )
        ModelRanksDict[pos][method] = model
#%%
## Load Models for Raw Player Stats
PlayerStatsModelDict= {"QB": {}, "RB": {}, "WR": {}, "TE": {}}
for pos in PlayerStatsModelDict.keys():
    for stat in ['pass_yards','pass_td','rush_yards','rush_td','rec','rec_yards','rec_td']:
        PlayerStatsModelDict[pos][stat]={}
        for method in ["BR","EN","NN","RF","GB","Tweedie"]:
            model = pickle.load(
                open(f"{etcdir}/model_pickles/{pos}_{method}_{stat}_model.pkl", "rb")
            )
            PlayerStatsModelDict[pos][stat][method]=model
    
    
    
    
    
    
    
    
    