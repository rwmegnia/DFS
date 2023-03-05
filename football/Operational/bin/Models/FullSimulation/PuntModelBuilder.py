#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:14:45 2022

@author: robertmegnia
"""

import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.dummy import DummyClassifier as DC
from sklearn.model_selection import train_test_split
import pickle
import os 
from datetime import datetime
import numpy as np
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir=f'{basedir}/ml_models'
def getClass(classes,probs):
    random=np.random.choice(np.arange(0.01,1.01,0.01))
    probs=probs-random
    if probs[probs>0].all():
        idx=np.where(probs==probs.max())[0][0]
        return classes[idx]
    probs=probs[probs>=0]
    idx=np.where(probs==probs.min())[0][0]
    return classes[idx]
# Builds Necessary Models for game simulation
pbp = nfl.import_pbp_data(range(2016,2022))
#%%
pbp['time']=pbp.time.apply(lambda x: datetime.strptime(x,'%M:%S') if x is not None else x)
pbp['end_time']=pbp.time.shift(-1)
pbp['elapsed_time']=(pbp.time-pbp.end_time).apply(lambda x: x.total_seconds())

punts=pbp[pbp.punt_attempt==1]
#
punts['end_territory']=punts.end_yard_line.apply(lambda x: x.split(' ')[0] if x not in [None,'50'] else x)
punts.loc[punts.end_territory==punts.defteam,'end_territory']='defteam'
punts.loc[punts.end_territory==punts.posteam,'end_territory']='posteam'
#
punts['start_territory']=punts.yrdln.apply(lambda x: x.split(' ')[0] if x not in [None,'50'] else x)
punts.loc[punts.start_territory==punts.defteam,'start_territory']='defteam'
punts.loc[punts.start_territory==punts.posteam,'start_territory']='posteam'

punts['end_yard_line']=punts.end_yard_line.apply(lambda x: int(x.split(' ')[1]) if x not in [None,'50'] else x)
punts['start_yard_line']=punts.yrdln.apply(lambda x: int(x.split(' ')[1]) if x not in [None,'50'] else x)

punts.end_yard_line.replace('50',50,inplace=True)
punts.start_yard_line.replace('50',50,inplace=True)
punts.loc[punts.end_territory=='defteam','end_yard_line']=100-punts.loc[punts.end_territory=='defteam','end_yard_line']
punts.loc[punts.start_territory=='defteam','start_yard_line']=100-punts.loc[punts.start_territory=='defteam','start_yard_line']

punts=punts[(punts.punt_blocked==1)|
            (punts.punt_fair_catch==1)|
            (punts.touchback==1)|
            (punts.fumble==1)|
            (punts.punt_out_of_bounds==1)|
            (punts.return_yards!=0)]
punts.loc[punts.punt_blocked==1,'punt_result']='blocked'
punts.loc[punts.punt_fair_catch==1,'punt_result']='fair_catch'
punts.loc[punts.touchback==1,'punt_result']='touchback'
punts.loc[(punts.fumble_lost==1)&(punts.return_yards<=0),'punt_result']='muffed'
punts.loc[punts.punt_out_of_bounds==1,'punt_result']='out_of_bounds'
punts.punt_result.fillna('returned',inplace=True)
#%%
punt_result_model=DC(strategy='stratified').fit(punts.qtr,punts.punt_result)

# Determiune Punt Result
pickle.dump(punt_result_model,open(f'{model_dir}/PuntModels/PuntResult_model.pkl','wb'))

# Determine Punt Distance/elapsed time
temp=punts[punts.kick_distance.isna()==False]
model=RF().fit(temp[['start_yard_line']],temp.kick_distance)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntDistance_model.pkl','wb'))
model=RF().fit(temp[(temp.kick_distance.isna()==False)&(temp.punt_result.isin(['fair_catch','touchback','out_of_bounds','muffed']))][['kick_distance']],temp[(temp.kick_distance.isna()==False)&(temp.punt_result.isin(['fair_catch','touchback','out_of_bounds','muffed']))].elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntNonReturn_elapsed_time_model.pkl','wb'))

# Punt Return Model/Elapsed Time
temp=punts[punts.punt_result=='returned']
model=DC(strategy='stratified').fit(temp[['qtr']],temp.return_yards)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntReturnYards_model.pkl','wb'))
temp=temp[temp.elapsed_time.isna()==False]
model=RF().fit(temp[['kick_distance','return_yards']],temp.elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntReturn_elapsed_time_model.pkl','wb'))

# Punt Return Fumble Lost
temp=temp[temp.fumble==1]
model=DC(strategy='stratified').fit(temp[['qtr']],temp.fumble_lost)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntReturnFumbleLost_model.pkl','wb'))
model=RF().fit(temp[['kick_distance','return_yards']],temp.elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntReturnFumbleLost_elapsed_time_model.pkl','wb'))

# Punt Def Fumble Recovery Touchdown
temp=temp[temp.fumble_lost==1]
model=DC(strategy='stratified').fit(temp[['qtr']],temp.return_touchdown)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntReturnDefFumbleTD_model.pkl','wb'))
model=RF().fit(temp[['kick_distance','return_yards']],temp.elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntReturnDefFumbleTD_elapsed_time_model.pkl','wb'))

#%%
# Punt Def Fumbel Recovery Yards
model=DC(strategy='stratified').fit(temp[['qtr']],temp.fumble_recovery_1_yards)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntReturnDefFumbleYards_model.pkl','wb'))
#%%
# Pre-Kick Penalty on Kicking Team/ElapsedTime
punts.loc[(punts.penalty==1)&
          (punts.end_yard_line<punts.start_yard_line)&
          (punts.penalty_team==punts.posteam)&
          (punts.kick_distance.isna()==True),'posteam_pre_kick_penalty']=1
temp=punts[punts.posteam_pre_kick_penalty==1]
model=DC(strategy='stratified').fit(temp[['qtr']],temp.elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntPosteamPreKickPenalty_elapsed_time_model.pkl','wb'))

# Pre-Kick Penalty on Receiving Team
punts.loc[(punts.penalty==1)&
          (punts.end_yard_line>punts.start_yard_line)&
          (punts.penalty_team==punts.defteam)&
          (punts.kick_distance.isna()==True),'defteam_pre_kick_penalty']=1
temp=punts[punts.defteam_pre_kick_penalty==1]
model=DC(strategy='stratified').fit(temp[['qtr']],temp.elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntDefteamPreKickPenalty_elapsed_time_model.pkl','wb'))

# Post-Kick Penalty on Kicking Team
punts.loc[(punts.penalty==1)&
          (punts.posteam_pre_kick_penalty!=1)&
          (punts.defteam_pre_kick_penalty!=1)&
          (punts.penalty_team==punts.posteam),'posteam_post_kick_penalty']=1

# Post-Kick Penalty on Receiving Team
punts.loc[(punts.penalty==1)&
          (punts.posteam_pre_kick_penalty!=1)&
          (punts.defteam_pre_kick_penalty!=1)&
          (punts.posteam_post_kick_penalty!=1)&
          (punts.penalty_team==punts.defteam),'defteam_post_kick_penalty']=1

punts.posteam_pre_kick_penalty.fillna(0,inplace=True)
punts.posteam_post_kick_penalty.fillna(0,inplace=True)
punts.defteam_pre_kick_penalty.fillna(0,inplace=True)
punts.defteam_post_kick_penalty.fillna(0,inplace=True)
#%%
model=DC(strategy='stratified').fit(punts[['qtr']],punts.penalty)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntPenalty_model.pkl','wb'))

temp=punts[punts.penalty==1]
temp.loc[temp.posteam_pre_kick_penalty==1,'punt_penalty_type']='posteam_pre_kick'
temp.loc[temp.posteam_post_kick_penalty==1,'punt_penalty_type']='posteam_post_kick'
temp.loc[temp.defteam_pre_kick_penalty==1,'punt_penalty_type']='defteam_pre_kick'
temp.loc[temp.defteam_post_kick_penalty==1,'punt_penalty_type']='defteam_post_kick'

model=DC(strategy='stratified').fit(temp[['qtr']],temp.punt_penalty_type)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntPenaltyType_model.pkl','wb'))

temp=punts[punts.posteam_pre_kick_penalty==1]
temp['penalty_string']=temp.penalty_type+' '+temp.penalty_yards.astype(str)
model=DC(strategy='stratified').fit(temp[['qtr']],temp.penalty_string)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntPosteamPreKickPenaltyType_model.pkl','wb'))

temp=punts[punts.posteam_post_kick_penalty==1]
temp['penalty_string']=temp.penalty_type+' '+temp.penalty_yards.astype(str)
temp=temp[temp.penalty_string.isna()==False]
model=DC(strategy='stratified').fit(temp[['qtr']],temp.penalty_string)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntPosteamPostKickPenaltyType_model.pkl','wb'))

temp=punts[punts.defteam_pre_kick_penalty==1]
temp['penalty_string']=temp.penalty_type+' '+temp.penalty_yards.astype(str)
model=DC(strategy='stratified').fit(temp[['qtr']],temp.penalty_string)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntDefteamPreKickPenaltyType_model.pkl','wb'))


temp=punts[punts.defteam_post_kick_penalty==1]
temp['penalty_string']=temp.penalty_type+' '+temp.penalty_yards.astype(str)
model=DC(strategy='stratified').fit(temp[['qtr']],temp.penalty_string)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntDefteamPostKickPenaltyType_model.pkl','wb'))
#%%
# Blocked Punt Result
temp=punts[(punts.punt_blocked==1)&(punts.penalty==0)]
temp.loc[temp.touchdown==1,'punt_block_result']='touchdown'
temp.loc[temp.safety==1,'punt_block_result']='safety'

# Determine who recovered block
temp['block_recovered_by']=temp.desc.str.extract('.*RECOVERED by ([A-Z]+)')
temp.loc[temp.block_recovered_by.isna()==True,'block_recovered_by']=temp.loc[temp.block_recovered_by.isna()==True].desc.str.extract('.*recovered by ([A-Z]+)').values
temp.loc[(temp.block_recovered_by.isna()==True)&(temp.punt_block_result.isna()==True),'punt_block_result']='punt_block_out_of_bounds'
temp.punt_block_result.fillna('block_returned',inplace=True)

# Determine where blocked punt was recovered
temp['block_recovered_at']=temp.desc.str.extract('.*RECOVERED by [A-Z]+-[0-9]+-[A-Za-z\'-.]+ at ([A-Z]+ [0-9]+)')
temp.loc[temp.block_recovered_at.isna()==True,'block_recovered_at']=temp.loc[temp.block_recovered_at.isna()==True].desc.str.extract('.*recovered by [A-Z]+-[0-9]+-[A-Za-z\'-.]+ at ([A-Z]+ [0-9]+)').values
temp.loc[(temp.block_recovered_at.isna()==True),'block_recovered_at']=temp.loc[(temp.block_recovered_at.isna()==True)].desc.str.extract('ball out of bounds at ([A-Z]+ [0-9]+)').values

# Determine where blocked punt was returned to
temp['block_returned_to']=temp.desc.str.extract('.*RECOVERED by [A-Z]+-[0-9]+-[A-Za-z\'-.]+ at [A-Z]+ [0-9]+. [0-9]+-[A-Za-z\'-.]+ to ([A-Z]+ [0-9]+)')
temp.loc[temp.block_returned_to.isna()==True,'block_returned_to']=temp.loc[temp.block_returned_to.isna()==True].desc.str.extract('.*recovered by [A-Z]+-[0-9]+-[A-Za-z\'-.]+ at [A-Z]+ [0-9]+. [0-9]+-[A-Za-z\'-.]+ to ([A-Z]+ [0-9]+)').values
temp.loc[(temp.block_returned_to.isna()==True)&(temp.punt_block_result=='block_returned'),'block_returned_to']=temp.loc[(temp.block_returned_to.isna()==True)&(temp.punt_block_result=='block_returned'),'block_recovered_at'].values

#
temp['territory']=temp.block_returned_to.apply(lambda x: x.split(' ')[0] if type(x)!=float else x)
temp['end_yard_line']=temp.block_returned_to.apply(lambda x: x.split(' ')[1] if type(x)!=float else x)
temp.end_yard_line=temp.end_yard_line.astype(float)
temp.loc[temp.territory==temp.defteam,'end_yard_line']=100-temp.loc[temp.territory==temp.defteam,'end_yard_line']
temp['block_return_yards']=temp.start_yard_line-temp.end_yard_line
##
model=RF().fit(temp[['start_yard_line']],temp.punt_block_result)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntBlockResult_model.pkl','wb'))

## Elapsed Time Models
model=RF().fit(temp[temp.punt_block_result=='touchdown'][['start_yard_line']],temp[temp.punt_block_result=='touchdown'].elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntBlockTD_elapsed_time_model.pkl','wb'))

model=RF().fit(temp[temp.punt_block_result=='safety'][['start_yard_line']],temp[temp.punt_block_result=='safety'].elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntBlockSafety_elapsed_time_model.pkl','wb'))

model=RF().fit(temp[temp.punt_block_result=='punt_block_out_of_bounds'][['start_yard_line']],temp[temp.punt_block_result=='punt_block_out_of_bounds'].elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntBlockOB_elapsed_time_model.pkl','wb'))

model=RF().fit(temp[temp.punt_block_result=='block_returned'][['start_yard_line','block_return_yards']],temp[temp.punt_block_result=='block_returned'].elapsed_time)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntBlockReturn_elapsed_time_model.pkl','wb'))

temp=temp[temp.punt_block_result=='block_returned']
model=RF().fit(temp[['punt_blocked']],temp.block_return_yards)
pickle.dump(model,open(f'{model_dir}/PuntModels/PuntBlockReturnYards_model.pkl','wb'))
