#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:55:15 2022

@author: robertmegnia
"""
import pickle
import os 
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir=f'{basedir}/ml_models'
# onside kick decision model
KickoffOption_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffOption_model.pkl','rb'))

# pre-kick penalty model
KickoffFrom_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffFrom_model.pkl','rb'))

# Kickoff Distance model
KickoffDistance_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffDistance_model.pkl','rb'))

# Kickoff Result Model
KickoffResult_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffResult_model.pkl','rb'))

#Kick Return Model
KickoffReturn_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffReturn_model.pkl','rb'))

# Determines if a penalty was comitted by kicking team
KickoffReturnedPenalty_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffReturnedPenalty_model.pkl','rb'))

# If a penalty was comitted by kicking team, what was the penalty
KickoffReturnedPenaltyDefteam_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffReturnedPenaltyDefteam_model.pkl','rb'))

# If kickoff result is a touchback, where is the end yard line?
# most of the time it will be the 25, but if there's a penalty 
# it will be a different spot, this model accounts for that
TouchbackEndYardLine_model = pickle.load(open(f'{model_dir}/KickoffModels/TouchbackEndYardLine_model.pkl','rb'))

# Fumble on kick return handlers
KickoffFumble_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffFumble_model.pkl','rb'))
KickoffFumbleLost_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffFumbleLost_model.pkl','rb'))
KickoffFumbleLostReturnTD_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffFumbleLostReturnTD_model.pkl','rb'))
KickoffFumbleLostReturnYards_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffFumbleLostReturnYards_model.pkl','rb'))

# Elapsed Time Handlers
KickoffReturn_elapsed_time_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffReturn_elapsed_time_model.pkl','rb'))
KickoffReturnTD_elapsed_time_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffReturnTD_elapsed_time_model.pkl','rb'))
KickoffReturnFumble_elapsed_time_model=pickle.load(open(f'{model_dir}/KickoffModels/KickoffReturnFumble_elapsed_time_model.pkl','rb'))


# Two Point Decision Handler
TwoPointDecision_model=pickle.load(open(f'{model_dir}/PATModels/TwoPointDecision_model.pkl','rb'))

#Extra Point Models
XPFrom_model=pickle.load(open(f'{model_dir}/PATModels/XPFrom_model.pkl','rb'))
XP_model = pickle.load(open(f'{model_dir}/PATModels/XP_model.pkl','rb'))
XPBlockedReturn_model=pickle.load(open(f'{model_dir}/PATModels/XPBlockedReturn_model.pkl','rb'))