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
    "Forward1": {},
    "Forward2": {},
    "Forward3": {},
    "Forward4": {},
    "Defenseman1": {},
    "Defenseman2": {},
    "Defenseman3": {},
    "Goalie1": {},
}
for pos in ModelDict.keys():
    for method in ["BR", "EN", "NN", "RF", "GB", "Tweedie"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_{method}_DKPts_model.pkl", "rb")
        )
        ModelDict[f"{pos}"][method] = model

# Load Player Share Models

# Load Models for Player DKPts Projections
SharesModelDict = {}
for pos in ModelDict.keys():
    if pos == "Goalie1":
        continue
    for stat in ["goals", "assists", "shots", "blocked", "DKPts"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_RF_{stat}_share_model.pkl", "rb")
        )
        SharesModelDict[f"{pos}_{stat}"] = model
# Load Team ML MOdels
TeamModelDict = {}
for method in ["BR", "EN", "RF", "GB", "Tweedie"]:
    for stat in ["goals", "assists", "shots", "blocked", "DKPts"]:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{stat}_{method}_Team_model.pkl", "rb")
        )
        TeamModelDict[f"{method}{stat}"] = model
