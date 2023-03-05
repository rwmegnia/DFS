#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 23:40:50 2022

@author: robertmegnia
"""

import pandas as pd
import nfl_data_py as nfl
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
import os
rosters=nfl.import_rosters(range(2016,2023))
depth_charts=nfl.import_depth_charts(range(2016,2023))
depth_charts.drop('depth_position',axis=1,inplace=True)
depth_charts.drop_duplicates(inplace=True)
snaps = nfl.import_snap_counts(range(2016,2022))
if os.path.exists('./pbp_data/2016_2022_pbp_data.csv')==False:
    temp=nfl.import_pbp_data(range(2016,2023))
    # Merge in positions of passers, rushers, and receivers
    temp = temp.merge(depth_charts.rename({'gsis_id':'passer_player_id',
                                           'club_code':'posteam',
                                           'position':'passer_position'},axis=1)[
                                               ['passer_player_id',
                                                'posteam',
                                                'week',
                                                'season',
                                                'passer_position']
                                               ],on=['posteam','week','season','passer_player_id'],how='left')
    temp.drop_duplicates(inplace=True)
    temp = temp.merge(depth_charts.rename({'gsis_id':'rusher_player_id',
                                           'club_code':'posteam',
                                           'position':'rusher_position'},axis=1)[
                                               ['rusher_player_id',
                                                'posteam',
                                                'week',
                                                'season',
                                                'rusher_position']
                                               ],on=['posteam','week','season','rusher_player_id'],how='left')
    temp.drop_duplicates(inplace=True)
    temp = temp.merge(depth_charts.rename({'gsis_id':'receiver_player_id',
                                           'club_code':'posteam',
                                           'position':'receiver_position'},axis=1)[
                                               ['receiver_player_id',
                                                'posteam',
                                                'week',
                                                'season',
                                                'receiver_position']
                                               ],on=['posteam','week','season','receiver_player_id'],how='left')
    temp.drop_duplicates(inplace=True)
else:
    temp=pd.read_csv('./pbp_data/2016_2022_pbp_data.csv')
#%%
pbp=temp[temp.n_offense.isna()==False]
pbp=pbp[pbp.play_type_nfl.isin(['RUSH','PASS','FIELD_GOAL','SACK','XP_KICK','PAT2'])]
pbp=pbp[(pbp.offense_personnel.str.contains('DL')==False)&
        (pbp.offense_personnel.str.contains('LB')==False)&
        (pbp.offense_personnel.str.contains('2 QB')==False)&
        (pbp.offense_personnel.str.contains('3 QB')==False)&
        (pbp.offense_personnel.str.contains('DB')==False)&
        (pbp.offense_personnel.str.contains('LS')==False)&
        (pbp.offense_personnel.str.contains('K')==False)&
        (pbp.offense_personnel.str.contains('P')==False)]
rushing = pbp[(pbp.rush_attempt==1)&(pbp.qb_dropback==0)]
rushing=rushing[(rushing.n_offense==11)&(rushing.n_defense==11)]
rushing=rushing[rushing.rusher_position.isin(['QB','RB','WR','FB'])]
#%%
rush_left = rushing[rushing.run_location=='left']
for i in range(1,12):
    rush_left[f'offense_player_{i}_id']=rush_left.offense_players.apply(lambda x: x.split(';')[i-1])
    rush_left = rush_left.merge(depth_charts.rename({'position':f'offense_player_{i}_position',
                                                     'club_code':'posteam',
                                                     'gsis_id':f'offense_player_{i}_id'},axis=1)[
                                                         [f'offense_player_{i}_id',
                                                          'week',
                                                          'season',
                                                          'posteam',
                                                          f'offense_player_{i}_position']
                                                         ],on=[f'offense_player_{i}_id','week','season','posteam'],how='left')
    rush_left.drop_duplicates(inplace=True)
    # rush_left[f'defense_player_{i}_id']=rush_left.defense_players.apply(lambda x: x.split(';')[i-1])
#%%
pbp_rush_left_frames=[]
for i in range(1,12):
    rush_left_frame=rush_left.groupby(['game_id','game_date',f'offense_player_{i}_id',f'offense_player_{i}_position','play_id','rusher_position','posteam']).mean()
    rush_left_frame=rush_left_frame.reset_index().rename({f'offense_player_{i}_id':'gsis_id',
                                                          f'offense_player_{i}_position':'position'},axis=1)
    pbp_rush_left_frames.append(rush_left_frame)

pbp_rush_left=pd.concat(pbp_rush_left_frames)
pbp_rush_left.position.replace({'G':'OL',
                                'C':'OL',
                                'T':'OL',
                                'FB':'OL',},inplace=True)
rb_rush_left=pbp_rush_left[pbp_rush_left.rusher_position=='RB'].groupby(['game_id','posteam','position','game_date']).mean()[['epa','rushing_yards','tackled_for_loss']]
# pbp_rush_left.reset_index(inplace=True)
#%%
rush_right = rushing[rushing.run_location=='right']
for i in range(1,12):
    rush_right[f'offense_player_{i}_id']=rush_right.offense_players.apply(lambda x: x.split(';')[i-1])
    rush_right = rush_right.merge(depth_charts.rename({'position':f'offense_player_{i}_position',
                                                     'club_code':'posteam',
                                                     'gsis_id':f'offense_player_{i}_id'},axis=1)[
                                                         [f'offense_player_{i}_id',
                                                          'week',
                                                          'season',
                                                          'posteam',
                                                          f'offense_player_{i}_position']
                                                         ],on=[f'offense_player_{i}_id','week','season','posteam'],how='right')
    rush_right.drop_duplicates(inplace=True)
    # rush_right[f'defense_player_{i}_id']=rush_right.defense_players.apply(lambda x: x.split(';')[i-1])
pbp_rush_right_frames=[]
for i in range(1,12):
    rush_right_frame=rush_right.groupby(['game_id','game_date',f'offense_player_{i}_id',f'offense_player_{i}_position','play_id','rusher_position','posteam']).mean()
    rush_right_frame=rush_right_frame.reset_index().rename({f'offense_player_{i}_id':'gsis_id',
                                                          f'offense_player_{i}_position':'position'},axis=1)
    pbp_rush_right_frames.append(rush_right_frame)

pbp_rush_right=pd.concat(pbp_rush_right_frames)
pbp_rush_right.position.replace({'G':'OL',
                                'C':'OL',
                                'T':'OL',
                                'FB':'OL',},inplace=True)
rb_rush_right=pbp_rush_right[pbp_rush_right.rusher_position=='RB'].groupby(['game_id','posteam','position','game_date']).mean()[['epa','rushing_yards','tackled_for_loss']]
rush_middle = rushing[rushing.run_location=='middle']
for i in range(1,12):
    rush_middle[f'offense_player_{i}_id']=rush_middle.offense_players.apply(lambda x: x.split(';')[i-1])
    rush_middle = rush_middle.merge(depth_charts.rename({'position':f'offense_player_{i}_position',
                                                     'club_code':'posteam',
                                                     'gsis_id':f'offense_player_{i}_id'},axis=1)[
                                                         [f'offense_player_{i}_id',
                                                          'week',
                                                          'season',
                                                          'posteam',
                                                          f'offense_player_{i}_position']
                                                         ],on=[f'offense_player_{i}_id','week','season','posteam'],how='middle')
    rush_middle.drop_duplicates(inplace=True)
    # rush_middle[f'defense_player_{i}_id']=rush_middle.defense_players.apply(lambda x: x.split(';')[i-1])
#%%
pbp_rush_middle_frames=[]
for i in range(1,12):
    rush_middle_frame=rush_middle.groupby(['game_id','game_date',f'offense_player_{i}_id',f'offense_player_{i}_position','play_id','rusher_position','posteam']).mean()
    rush_middle_frame=rush_middle_frame.reset_index().rename({f'offense_player_{i}_id':'gsis_id',
                                                          f'offense_player_{i}_position':'position'},axis=1)
    pbp_rush_middle_frames.append(rush_middle_frame)

pbp_rush_middle=pd.concat(pbp_rush_middle_frames)
pbp_rush_middle.position.replace({'G':'OL',
                                'C':'OL',
                                'T':'OL',
                                'FB':'OL',},inplace=True)
rb_rush_middle=pbp_rush_middle[pbp_rush_middle.rusher_position=='RB'].groupby(['game_id','posteam','position','game_date']).mean()[['epa','rushing_yards','tackled_for_loss']]
# pbp_rush_middle.reset_index(inplace=True)
