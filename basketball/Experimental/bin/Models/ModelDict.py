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
    "starter": {'C':{},
                'SG':{},
                'PG':{},
                'SF':{},
                'PF':{}},
    "bench": {'C':{},
                'SG':{},
                'PG':{},
                'SF':{},
                'PF':{}},
}
for starter in ModelDict.keys():
    for pos in ['C','SG','PG','SF','PF']:
        for method in ["EN", "RF", "GB"]:
            model = pickle.load(
                open(f"{etcdir}/model_pickles/{method}_{pos}_dkpts_{starter}_model.pkl", "rb")
            )
            ModelDict[starter][pos][method] = model

# Load Player Share Models

SharesModelDict = {'RF':{'C':{},
                        'SG':{},
                        'PG':{},
                        'SF':{},
                        'PF':{}},
                   'GB':{'C':{},
                         'SG':{},
                         'PG':{},
                         'SF':{},
                         'PF':{}},
                   }
for method in SharesModelDict.keys():
    for pos in ['C','SG','PG','SF','PF']:
        for starter in ['starter','bench']:
            for stat in ["pts","fg3m","ast","reb","stl","blk","tov","dkpts"]:
                model = pickle.load(
                    open(f"{etcdir}/model_pickles/{method}_{pos}_pct_{stat}_{starter}_model.pkl", "rb")
                )
                SharesModelDict[method][pos][f"{starter}_{stat}"] = model

# Load Player Stat Models

StatsModelDict =  {
                   'EN':{'C':{},
                         'SG':{},
                         'PG':{},
                         'SF':{},
                         'PF':{}},
                   'RF':{'C':{},
                         'SG':{},
                         'PG':{},
                         'SF':{},
                         'PF':{}},
                   'GB':{'C':{},
                         'SG':{},
                         'PG':{},
                         'SF':{},
                         'PF':{}},
                   }
for method in StatsModelDict.keys():
    for pos in ['C','SG','PG','SF','PF']:
        for starter in ['starter','bench']:
            for stat in ["pts","fg3m","ast","reb","stl","blk","to","dkpts"]:
                model = pickle.load(
                    open(f"{etcdir}/model_pickles/{method}_{pos}_{stat}_{starter}_model.pkl", "rb")
                )
                StatsModelDict[method][pos][f"{starter}_{stat}"] = model
# Load Team ML MOdels
TeamModelDict = {  'RF':{},
                   'GB':{},}
for method in ["RF", "GB"]:
    for stat in ["pts","fg3m","ast","reb","stl","blk","to",'dkpts']:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{stat}_{method}_Team_model.pkl", "rb")
        )
        TeamModelDict[method][f"{method}{stat}"] = model
        
# Load MInutes Model
MinsModelDict={'True':{},
               'False':{}}
for starter in ["True","False"]:
    for method in ['BR','EN','NN','RF','GB','Tweedie']:
        model = pickle.load(
            open(f"{etcdir}/model_pickles/{method}_{starter}_minutes_model.pkl", "rb")
        )
        MinsModelDict[starter][method]=model
    
