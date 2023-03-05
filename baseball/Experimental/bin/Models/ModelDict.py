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

# Load Models for Player DKPts Projection
ModelDict = {
    "Batter_LvsL": {},
    "Batter_LvsR": {},
    "Batter_RvsL": {},
    "Batter_RvsR": {},
    "Pitcher_L": {},
    "Pitcher_R": {},
}
for pos in ModelDict.keys():
    if 'Pitcher' in pos:
        methods= ["BR","EN","NN", "RF", "GB","Tweedie"]
    else:
        methods= ['BR',"EN",'NN', "RF", "GB",'Tweedie']
    for method in methods:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{pos}_{method}_DKPts_model.pkl", "rb")
        )
        ModelDict[f"{pos}"][method] = model


