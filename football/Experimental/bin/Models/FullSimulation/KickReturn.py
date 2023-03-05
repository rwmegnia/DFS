#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:12:33 2023

@author: robertmegnia
"""

import pandas as pd
import numpy as np
from SimulationModels import AirYardsModel,shortPassModel,mediumPassModel,deepPassModel

def getClass(classes, probs):
    classes = classes[probs > 0]
    probs = probs[probs > 0]
    random = np.random.choice(np.arange(0.00001, 1.01, 0.00001))
    probs = probs.cumsum() - random
    try:
        idx = list(probs).index(min([i for i in list(probs) if i >= 0]))
    except ValueError:
        idx = np.where(probs == probs.max())[0][0]
    return classes[idx]

def selectOffensivePlayers(game, position, n):
    df = pd.concat(game.posteam.depth_chart[position])
    df.drop_duplicates(inplace=True)
    df = df.groupby("gsis_id", as_index=False).first()
    df.sort_values(by="depth_team", inplace=True)
    if position=='QB':
        return df[df.depth_team==1]
    df.ewm_offense_pct.fillna(0,inplace=True)
    probs=(df.ewm_offense_pct/df.ewm_offense_pct.sum()).values
    if n>len(df):
        n=len(df)
    players = np.random.choice(
        df.gsis_id,replace=False,p=probs,size=n)
    return df[df.gsis_id.isin(players)]

def selectDefensivePlayers(game, position, n):
    df = pd.concat(position)
    df.drop_duplicates(inplace=True)
    df = df.groupby("gsis_id", as_index=False).first()
    df.sort_values(by="depth_team", inplace=True)
    df.ewm_defense_pct.fillna(0,inplace=True)
    probs=(df.ewm_defense_pct/df.ewm_defense_pct.sum()).values
    if n>len(df):
        n=len(df)
    players = np.random.choice(
        df.gsis_id,replace=False,p=probs,size=n)
    return df[df.gsis_id.isin(players)]
    
    
def determineRunner(game):
    potential_runners = pd.concat([game.RB,game.QB,game.WR])
    potential_runners=potential_runners[potential_runners.rush_share.isna()==False]
    probs=(potential_runners.rush_share/potential_runners.rush_share.sum()).values
    runner=np.random.choice(potential_runners.gsis_id,p=probs)
    runner = potential_runners[potential_runners.gsis_id==runner]
    runpos = runner.position.values[0]
    return runner,runpos
    
def determineReceiver(game,pass_length):
    potential_receivers = pd.concat([game.TE,game.RB,game.WR])
    potential_receivers=potential_receivers[potential_receivers[f'rec_ewm_{pass_length}_target_share'].isna()==False]
    probs=(potential_receivers[f'rec_ewm_{pass_length}_target_share']/
           potential_receivers[f'rec_ewm_{pass_length}_target_share'].sum()).values
    receiver=np.random.choice(potential_receivers.gsis_id,p=probs)
    receiver = potential_receivers[potential_receivers.gsis_id==receiver]
    return receiver

def retrieveAirYardsFeatures(game):
    qb_data = game.QB[['qb_ewm_air_epa',
                       'qb_ewm_air_yards_per_attempt',
                       'qb_ewm_air_yards_efficiency']].mean().to_frame().T
    dl_data = game.DL[['def_air_epa',
                       'def_air_yards_per_attempt',
                       'def_air_yards_efficiency']].mean().to_frame().T
    lb_data = game.LB[['def_air_epa',
                       'def_air_yards_per_attempt',
                       'def_air_yards_efficiency']].mean().to_frame().T
    db_data = game.DB[['def_air_epa',
                       'def_air_yards_per_attempt',
                       'def_air_yards_efficiency']].mean().to_frame().T
    def_data=pd.concat([dl_data,lb_data,db_data]).mean().to_frame().T
    features = pd.DataFrame(
        {
            'qb_ewm_air_epa':qb_data.qb_ewm_air_epa.values[0],
            'qb_ewm_air_yards_per_attempt':qb_data.qb_ewm_air_yards_per_attempt.values[0],
            'qb_ewm_air_yards_efficiency':qb_data.qb_ewm_air_yards_efficiency,
            'def_air_epa':def_data.def_air_epa.values[0],
            'def_air_yards_per_attempt':def_data.def_air_yards_per_attempt.values[0],
            'def_air_yards_efficiency':def_data.def_air_yards_efficiency.values[0],
            'down':[game.down],
            'ydstogo':[game.ydstogo],
            'yardline':[game.yardline],
            'score_differential':[game.score_differential],
            'half_seconds_remaining':[game.half_seconds_remaining],
            'pass_rushers':[game.n_pass_rushers]})
    return features

def determinePassLength(game):
    air_yards_features = retrieveAirYardsFeatures(game)
    air_yards_probs = AirYardsModel.predict_proba(air_yards_features)[0]
    air_yards = getClass(AirYardsModel.classes_,air_yards_probs)
    if air_yards<8:
        pass_length='short'
    elif (air_yards>=8)&(air_yards<15):
        pass_length='medium'
    else:
        pass_length='deep'
    return air_yards,pass_length
    
def retrieveYacFeatures(game,air_yards):
    rec_data = game.receiver[['rec_ewm_epa','rec_ewm_yac_epa','rec_ewm_yards_after_catch']].mean().to_frame().T
    dl_data = game.DL[['def_epa','def_yac_epa','def_yards_after_catch']].mean().to_frame().T
    lb_data = game.LB[['def_epa','def_yac_epa','def_yards_after_catch']].mean().to_frame().T
    db_data = game.DB[['def_epa','def_yac_epa','def_yards_after_catch']].mean().to_frame().T
    def_data = pd.concat([dl_data,lb_data,db_data]).mean().to_frame().T    
    game_feats=pd.DataFrame({'down':[game.down],
                             'ydstogo':[game.ydstogo],
                             'yardline':[game.yardline],
                             'distance_to_sticks':[game.ydstogo-air_yards],
                             'air_yards':[air_yards],
                             'number_of_pass_rushers':[game.n_pass_rushers]})
    features = pd.concat([rec_data,def_data,game_feats],axis=1)
    return features
    
def retrievePassFeatures(game,pass_length,air_yards):
    ol_data = game.OL[['OL_epa',f'OL_{pass_length}_target_epa']].mean().to_frame().T
    qb_data = game.QB[['qb_ewm_epa',f'qb_ewm_{pass_length}_pass_epa',
                       'qb_ewm_air_yards_efficiency','qb_ewm_completion_percentage',
                       f'qb_ewm_{pass_length}_completion_percentage',
                       'qb_ewm_int_rate','qb_ewm_passer_rating']].mean().to_frame().T
    rec_data = game.receiver[['rec_ewm_epa',f'rec_ewm_{pass_length}_target_epa',
                              'rec_ewm_air_yards_efficiency','rec_ewm_catch_rate',
                              f'rec_ewm_{pass_length}_catch_rate']].mean().to_frame().T
    dl_data = game.DL[['def_epa',f'def_{pass_length}_target_epa','def_catch_rate',
                       f'def_{pass_length}_catch_rate','def_air_yards_efficiency']].mean().to_frame().T
    lb_data = game.LB[['def_epa',f'def_{pass_length}_target_epa','def_catch_rate',
                       f'def_{pass_length}_catch_rate','def_air_yards_efficiency']].mean().to_frame().T
    db_data = game.DB[['def_epa',f'def_{pass_length}_target_epa','def_catch_rate',
                       f'def_{pass_length}_catch_rate','def_air_yards_efficiency']].mean().to_frame().T
    def_data = pd.concat([dl_data,lb_data,db_data]).mean().to_frame().T
    game_feats=pd.DataFrame({'down':[game.down],
                             'ydstogo':[game.ydstogo],
                             'yardline':[game.yardline],
                             'distance_to_sticks':[game.ydstogo-air_yards],
                             'air_yards':[air_yards],
                             'number_of_pass_rushers':[game.n_pass_rushers]})
    features = pd.concat([ol_data,qb_data,rec_data,def_data,game_feats],axis=1)
    return features

def retrieveRushFeatures(game,runner):
    defenders_in_box = game.defenders_in_box
    ol_data = game.OL[['line_epa','adjusted_line_yards']].mean().to_frame().T
    dl_data = game.DL[['def_line_epa','def_adjusted_line_yards']].mean().to_frame().T
    lb_data = game.LB[['def_second_level_rushing_yards']].mean().to_frame().T
    db_data =game.DB[['def_open_field_rushing_yards']].mean().to_frame().T
    runner_data = runner[['ypc','open_field_ypc','rush_epa']].reset_index(drop=True)
    features =pd.concat([runner_data,ol_data,dl_data,lb_data,db_data],axis=1)
    features['defenders_in_box']=defenders_in_box
    return features
    
def retrieveSackFeatures(game):
    pass_rushers = game.n_pass_rushers
    qb_data = game.QB[['qb_sack_rate']].mean().to_frame().T.values[0]
    ol_data = game.OL[['sack_allowed_rate']].mean().to_frame().T.values[0]
    dl_data = game.DL[['sack_rate']].mean().to_frame().T
    lb_data = game.LB[['sack_rate']].mean().to_frame().T
    def_data=pd.concat([dl_data,lb_data]).mean().values[0]
    sack_features =pd.DataFrame({'sack_allowed_rate':ol_data,
                                 'qb_sack_rate':qb_data,
                                 'sack_rate':def_data,
                                 'down':[game.down],
                                 'ydstogo':[game.ydstogo],
                                 'yardline':[game.yardline],
                                 'n_pass_rushers':[pass_rushers],
                                 'score_diff':[game.score_differential],
                                 'seconds':[game.half_seconds_remaining]})
    return sack_features
    
def stat_frame(game):
    return pd.DataFrame({
        'rush_attempt':[],
        'pass_attempt':[],
        'rusher_player_id':[],
        'passer_player_id':[],
        'receiver_player_id':[],
        'passing_yards':[],
        'rushing_yards':[],
        'pass_touchdown':[],
        'rush_touchdown':[]})

def getPassModel(game):
    if game.pass_length=='short':
        return shortPassModel
    elif game.pass_length == 'medium':
        return mediumPassModel
    else:
        return deepPassModel