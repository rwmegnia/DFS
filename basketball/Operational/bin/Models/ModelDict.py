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
ModelDict = {
    "starter": {},
    "bench": {},
}
for starter in ModelDict.keys():
    for method in ["BR", "EN", "NN", "RF", "GB", "Tweedie"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{method}_DKPts_{starter}_model.pkl", "rb")
        )
        ModelDict[starter][method] = model

# Load Player Share Models

SharesModelDict = {'RF':{},
                   'GB':{},}
for method in SharesModelDict.keys():
    for starter in ['starter','bench']:
        for stat in ["pts","fg3m","ast","reb","stl","blk","tov","dkpts"]:
            model = pickle.load(
                open(f"{etcdir}/model_pickles/{method}_pct_{stat}_{starter}_model.pkl", "rb")
            )
            SharesModelDict[method][f"{starter}_{stat}"] = model
# Load Team ML MOdels
TeamModelDict = {  'RF':{},
                   'GB':{},}
for method in ["RF", "GB"]:
    for stat in ["pts","fg3m","ast","reb","stl","blk","to",'dkpts']:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{stat}_{method}_Team_model.pkl", "rb")
        )
        TeamModelDict[method][f"{method}{stat}"] = model
