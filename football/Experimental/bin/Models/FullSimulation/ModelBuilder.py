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
import datetime
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir=f'{basedir}/ml_models'
#%%
# Builds Necessary Models for game simulation
pbp = nfl.import_pbp_data(range(2017,2022))
pbp['time']=pbp.time.apply(lambda x: datetime.strptime(x,'%M:%S'))
pbp['end_time']=pbp.time.shift(-1)
pbp['elapsed_time']=(pbp.time-pbp.end_time).apply(lambda x: x.total_seconds())
# Determine if clock if stopped or running
pbp['clock_running'] = True

# Stop clock for incomplete pass
pbp.loc[(pbp.complete_pass==0)&
        (pbp.sack==0)&
        (pbp.play_type!='run'),'clock_running']=False

# Stop clock for running out of bounds play with 2 minutes left in first half
pbp.loc[(pbp.qtr==2)&
        (pbp.half_seconds_remaining<=120)&
        (pbp.play_type=='run')&
        (pbp.out_of_bounds==1),'clock_running']=False

# Stop clock for running out of bounds play with 5 minutes left in second half
pbp.loc[(pbp.qtr==4)&
        (pbp.half_seconds_remaining<=300)&
        (pbp.play_type=='run')&
        (pbp.out_of_bounds==1),'clock_running']=False
# Stop clock for receiving out of bounds play with 2 minutes left in first half
pbp.loc[(pbp.qtr==2)&
        (pbp.half_seconds_remaining<=120)&
        (pbp.complete_pass==1)&
        (pbp.out_of_bounds==1),'clock_running']=False
# Stop clock for receiving out of bounds play with 5 minutes left in second half
pbp.loc[(pbp.qtr==4)&
        (pbp.half_seconds_remaining<=300)&
        (pbp.complete_pass==1)&
        (pbp.out_of_bounds==1),'clock_running']=False

# Stop clock after any field goal attempt
pbp.loc[(pbp.field_goal_attempt==1),'clock_running']=False

# Stop clock after any punt
pbp.loc[(pbp.punt_attempt==1),'clock_running']=False

# Stop clock after any 4th down that doesn't end with a first down
pbp.loc[(pbp.down==4)&(pbp.first_down==0),'clock_running']=False

# Stop clock for a timeout
pbp.loc[(pbp.timeout==1),'clock_running']=False

# Stop clock after a score or turnover
pbp.loc[(pbp.touchdown==1)|
        (pbp.safety==1)|
        (pbp.interception==1)|
        (pbp.fumble_lost==1),'clock_running']=False

# Stop clock for kickoff
pbp.loc[pbp.play_type.isin(['kickoff',None]),'clock_running']=False
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

#%% If fumble wasn't recovered for TD, how many yards did the defense recover for?
fumble_returned=fumble_lost[fumble_lost.fumble_lost==True]
model=DC(strategy='stratified').fit(fumble_returned.qtr,fumble_returned.fumble_recovery_1_yards)
pickle.dump(model,open(f'{model_dir}/KickoffFumbleLostReturnYards_model.pkl','wb'))


