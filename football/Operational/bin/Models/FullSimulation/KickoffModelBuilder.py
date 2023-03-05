#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 18:52:10 2022

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
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir=f'{basedir}/ml_models'
#%%
# Builds Necessary Models for game simulation
pbp = nfl.import_pbp_data(range(2017,2022))
pbp['time']=pbp.time.apply(lambda x: datetime.strptime(x,'%M:%S') if x is not None else x)
pbp['end_time']=pbp.time.shift(-1)
pbp['elapsed_time']=(pbp.time-pbp.end_time).apply(lambda x: x.total_seconds())
#%%
kickoffs = pbp[(pbp.kickoff_attempt==1)]
kickoffs['score_differential']=kickoffs.defteam_score-kickoffs.posteam_score
kickoffs.loc[kickoffs.penalty_team==kickoffs.posteam,'penalty_team']='posteam'
kickoffs.loc[kickoffs.penalty_team==kickoffs.defteam,'penalty_team']='defteam'
kickoffs.loc[kickoffs.desc.str.contains('kicks onside')==True,'onside_kick']=1
kickoffs.onside_kick.fillna(0,inplace=True)
kickoffs.end_yard_line.fillna(0,inplace=True)

#%% Build Kickoff Option Model - Onside or Regular
# 
# Only consider onside kick with less than 5 minutes in the game

temp = kickoffs[kickoffs.game_seconds_remaining<=300]

# train with 125 random samples of an onside kick and regular kick
train = pd.concat([temp[temp.onside_kick==1].sample(125),
                    temp[temp.onside_kick==0].sample(125)])

model=RF().fit(train[['game_seconds_remaining',
                      'score_differential',
                      'defteam_timeouts_remaining']],train.onside_kick)
pickle.dump(model,open(f'{model_dir}/KickoffOption_model.pkl','wb'))

#%% Build KickoffFrom Model 
kickoffs['kickoff_from']=kickoffs.yrdln.apply(lambda x: int(x.split(' ')[1]))

# Remove Opening Kickoff and OT opening Kickoff
temp=kickoffs[kickoffs.game_seconds_remaining!=3600]
temp.drop(temp.loc[(temp.qtr==5)&(temp.game_seconds_remaining==600)].index,inplace=True)
model = DC(strategy='stratified').fit(temp.qtr,temp.kickoff_from)
pickle.dump(model,open(f'{model_dir}/KickoffFrom_model.pkl','wb'))
#%% Build Kickoff Result Model Limiting Possible Results to touchback, return, or out of bounds
#
temp = kickoffs[(kickoffs.onside_kick==0)&(kickoffs.kickoff_fair_catch==0)]
temp.loc[temp.touchback==1,'kickoff_result']='touchback'
temp.loc[temp.kickoff_out_of_bounds==1,'kickoff_result']='out_of_bounds'
temp.loc[(temp.penalty==1)&(temp.penalty_type.isin(['Offside on Free Kick',
                                                    'Illegal Formation',
                                                    'Delay of Kickoff',
                                                    'Delay of Game'])),'kickoff_result']='pre_kick_penalty'
temp.kickoff_result.fillna('returned',inplace=True)
temp=temp[temp.kickoff_result!='pre_kick_penalty']
model = DC(strategy='stratified').fit(temp.qtr,temp.kickoff_result)
pickle.dump(model,open(f'{model_dir}/KickoffResult_model.pkl','wb'))
#%% Build Kick Return Model - exclude lateral returns
#
temp = temp[(temp.kickoff_result=='returned')&(temp.lateral_return==0)]

# Remove Fumbles returned for touchdowns
temp.drop(temp[(temp.return_touchdown==1)&(temp.fumble==1)].index,inplace=True)

# Determine which territory the play ended in (return team or kicking team (posteam/defteam))
temp.loc[~temp.end_yard_line.isin([0,'50']),'end_yard_line_territory']=temp.loc[~temp.end_yard_line.isin([0,'50'])].end_yard_line.apply(lambda x: x.split(' ')[0])
temp.loc[temp.end_yard_line_territory==temp.posteam,'end_yard_line_territory']='posteam'
temp.loc[temp.end_yard_line_territory==temp.defteam,'end_yard_line_territory']='defteam'
temp.loc[temp.end_yard_line_territory==0,'end_yard_line_territory']=0
temp.loc[temp.end_yard_line_territory=='50','end_yard_line_territory']='50'
temp.loc[~temp.end_yard_line.isin([0,'50']),'end_yard_line']=temp.loc[~temp.end_yard_line.isin([0,'50'])].end_yard_line.apply(lambda x: int(x.split(' ')[1]))
temp.loc[temp.end_yard_line=='50','end_yard_line']=50
temp.loc[temp.end_yard_line_territory=='defteam','end_yard_line']=100-temp.loc[temp.end_yard_line_territory=='defteam','end_yard_line']
temp.loc[temp.end_yard_line==0,'end_yard_line']=100
temp['returned_from']=temp.end_yard_line-temp.return_yards
temp.loc[temp.penalty_team=='defteam','returned_from']=temp.loc[temp.penalty_team=='defteam','end_yard_line']-temp.loc[temp.penalty_team=='defteam',['return_yards','penalty_yards']].sum(axis=1)

temp.kickoff_from=100-temp.kickoff_from
temp['kickoff_distance']=temp.kickoff_from-temp.returned_from
## Fit Kickoff Distance
model=DC(strategy='stratified').fit(temp.kickoff_from,temp.kickoff_distance)
pickle.dump(model,open(f'{model_dir}/KickoffDistance_model.pkl','wb'))

## Fit Return Yards Model
model=DC(strategy='stratified').fit(temp.returned_from,temp.return_yards)
pickle.dump(model,open(f'{model_dir}/KickoffReturn_model.pkl','wb'))


##Fit Kickoff Penalty Models
temp = kickoffs[(kickoffs.onside_kick==0)&(kickoffs.kickoff_fair_catch==0)]
temp.loc[temp.touchback==1,'kickoff_result']='touchback'
temp.loc[temp.kickoff_out_of_bounds==1,'kickoff_result']='out_of_bounds'
temp.loc[(temp.penalty==1)&(temp.penalty_type.isin(['Offside on Free Kick',
                                                    'Illegal Formation',
                                                    'Delay of Kickoff',
                                                    'Delay of Game'])),'kickoff_result']='pre_kick_penalty'
temp.kickoff_result.fillna('returned',inplace=True)
temp = temp[temp.kickoff_result.isin(['returned','touchback'])]
#%%
touchback=temp[temp.kickoff_result=='touchback']
touchback['end_yard_line']=touchback.end_yard_line.apply(lambda x: int(x.split(' ')[1]))
model=DC(strategy='stratified').fit(touchback.qtr,touchback.end_yard_line)

pickle.dump(model,open(f'{model_dir}/TouchbackEndYardLine_model.pkl','wb'))
#%%
# Kickoff Return Defense Penalty Model
temp = kickoffs[(kickoffs.onside_kick==0)&(kickoffs.kickoff_fair_catch==0)]
temp.loc[temp.touchback==1,'kickoff_result']='touchback'
temp.loc[temp.kickoff_out_of_bounds==1,'kickoff_result']='out_of_bounds'
temp.loc[(temp.penalty==1)&(temp.penalty_type.isin(['Offside on Free Kick',
                                                    'Illegal Formation',
                                                    'Delay of Kickoff',
                                                    'Delay of Game'])),'kickoff_result']='pre_kick_penalty'
temp.kickoff_result.fillna('returned',inplace=True)
returned=temp[temp.kickoff_result=='returned']
model=DC(strategy='stratified').fit(returned.qtr,returned.penalty)
pickle.dump(model,open(f'{model_dir}/KickoffReturnedPenalty_model.pkl','wb'))
#%%
# Kickoff returned Penalty Team Model
returned_penalty = returned[returned.penalty==1]
returned_penalty.loc[returned_penalty.penalty_team==returned_penalty.defteam,'penalty_team']='defteam'
returned_penalty.loc[returned_penalty.penalty_team==returned_penalty.posteam,'penalty_team']='posteam'
returned_penalty=returned_penalty[returned_penalty.penalty_team=='defteam']
# defteam penalty on return type model
returned_defteam_penalty = returned_penalty[returned_penalty.penalty_team=='defteam']
returned_defteam_penalty['penalty_type_yards']=returned_defteam_penalty.penalty_type+' '+returned_defteam_penalty.penalty_yards.astype(str)
model=DC(strategy='stratified').fit(returned_defteam_penalty.qtr,returned_defteam_penalty.penalty_type_yards)
pickle.dump(model,open(f'{model_dir}/KickoffReturnedPenaltyDefteam_model.pkl','wb'))
#%% Kickoff Fumbled Model
temp = kickoffs[(kickoffs.onside_kick==0)&(kickoffs.kickoff_fair_catch==0)]
temp.loc[temp.touchback==1,'kickoff_result']='touchback'
temp.loc[temp.kickoff_out_of_bounds==1,'kickoff_result']='out_of_bounds'
temp.loc[(temp.penalty==1)&(temp.penalty_type.isin(['Offside on Free Kick',
                                                    'Illegal Formation',
                                                    'Delay of Kickoff',
                                                    'Delay of Game'])),'kickoff_result']='pre_kick_penalty'
temp.kickoff_result.fillna('returned',inplace=True)
fumbled=temp[temp.kickoff_result=='returned']
# Was there a fumble?
model=DC(strategy='stratified').fit(fumbled.qtr,fumbled.fumble)
pickle.dump(model,open(f'{model_dir}/KickoffFumble_model.pkl','wb'))
#%%
# Who recovered fumble?
fumble_lost=fumbled[fumbled.fumble==1]
model=DC(strategy='stratified').fit(fumbled.qtr,fumbled.fumble_lost)
pickle.dump(model,open(f'{model_dir}/KickoffFumbleLost_model.pkl','wb'))
#%% If defense recovered, was it returned for a tuddy?
fumble_returned=fumble_lost[fumble_lost.fumble_lost==True]
model=DC(strategy='stratified').fit(fumble_returned.qtr,fumble_returned.return_touchdown)
pickle.dump(model,open(f'{model_dir}/KickoffFumbleLostReturnTD_model.pkl','wb'))
model=DC(strategy='stratified').fit(fumble_returned[fumble_returned.return_touchdown==1].qtr,fumble_returned[fumble_returned.return_touchdown==1].elapsed_time)
pickle.dump(model,open(f'{model_dir}/KickoffFumbleLostReturnTD_elapsed_time_model.pkl','wb'))

#%% If fumble wasn't recovered for TD, how many yards did the defense recover for?
fumble_returned=fumble_lost[fumble_lost.fumble_lost==True]
model=DC(strategy='stratified').fit(fumble_returned.qtr,fumble_returned.fumble_recovery_1_yards)
pickle.dump(model,open(f'{model_dir}/KickoffFumbleLostReturnYards_model.pkl','wb'))
#%%
elapsed_time_model=RF().fit(temp[['return_yards','randomizer']])

