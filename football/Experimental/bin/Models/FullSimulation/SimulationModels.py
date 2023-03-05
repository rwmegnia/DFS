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

penalty_dict={
 'Chop Block': 15.0,
 'Clipping': 15.0,
 'Defensive 12 On-field': 5.0,
 'Defensive Delay of Game': 5.0,
 'Defensive Holding': 5.0,
 'Defensive Offside': 5.0,
 'Defensive Pass Interference': 5.0,
 'Defensive Too Many Men on Field': 5.0,
 'Delay of Game': 5.0,
 'Delay of Kickoff': 5.0,
 'Disqualification': 15.0,
 'Encroachment': 5.0,
 'Face Mask': 15.0,
 'Fair Catch Interference': 15.0,
 'False Start': 5.0,
 'Horse Collar Tackle': 15.0,
 'Illegal Bat': 10.0,
 'Illegal Blindside Block': 15.0,
 'Illegal Block Above the Waist': 10.0,
 'Illegal Contact': 5.0,
 'Illegal Crackback': 15.0,
 'Illegal Double-Team Block': 10.0,
 'Illegal Formation': 5.0,
 'Illegal Forward Pass': 5.0,
 'Illegal Kick/Kicking Loose Ball': 10.0,
 'Illegal Motion': 5.0,
 'Illegal Peelback': 15.0,
 'Illegal Shift': 5.0,
 'Illegal Substitution': 5.0,
 'Illegal Touch Kick': 5.0,
 'Illegal Touch Pass': 5.0,
 'Illegal Use of Hands': 5.0,
 'Ineligible Downfield Kick': 5.0,
 'Ineligible Downfield Pass': 5.0,
 'Intentional Grounding': 10.0,
 'Interference with Opportunity to Catch': 15.0,
 'Invalid Fair Catch Signal': 5.0,
 'Kick Catch Interference': 15.0,
 'Leaping': 15.0,
 'Leverage': 15.0,
 'Low Block': 15.0,
 'Lowering the Head to Initiate Contact': 15.0,
 'Neutral Zone Infraction': 5.0,
 'Offensive 12 On-field': 5.0,
 'Offensive Holding': 10.0,
 'Offensive Offside': 5.0,
 'Offensive Pass Interference': 10.0,
 'Offensive Too Many Men on Field': 5.0,
 'Offside on Free Kick': 5.0,
 'Player Out of Bounds on Kick': 5.0,
 'Player Out of Bounds on Punt': 5.0,
 'Roughing the Kicker': 15.0,
 'Roughing the Passer': 15.0,
 'Running Into the Kicker': 5.0,
 'Short Free Kick': 5.0,
 'Taunting': 15.0,
 'Tripping': 10.0,
 'Unnecessary Roughness': 15.0,
 'Unsportsmanlike Conduct': 15.0
 }
auto_first_down=[ 'Defensive Holding', 
                 'Face Mask', 
                 'Horse Collar Tackle', 
                 'Illegal Contact',
                 'Illegal Use of Hands', 
                 'Lowering the Head to Initiate Contact', 
                 'Roughing the Passer',
                 'Taunting', 
                 'Unnecessary Roughness', 
                 'Unsportsmanlike Conduct']
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

#Field Goal Elapsed Time
FG_ElapsedTime = pickle.load(open(f'{model_dir}/FieldGoalModels/FG_ElapsedTime_model.pkl','rb'))
# Punt Models
punt_models = [
 'PuntBlockOB_elapsed_time_model.pkl',
 'PuntBlockRecoveredAt_model.pkl',
 'PuntBlockRecoveredBy_model.pkl',
 'PuntBlockResult_model.pkl',
 'PuntBlockReturn_elapsed_time_model.pkl',
 'PuntBlockReturnYards_model.pkl',
 'PuntBlockSafety_elapsed_time_model.pkl',
 'PuntBlockTD_elapsed_time_model.pkl',
 'PuntDefteamPostKickPenaltyType_model.pkl',
 'PuntDefteamPreKickPenalty_elapsed_time_model.pkl',
 'PuntDefteamPreKickPenaltyType_model.pkl',
 'PuntDistance_model.pkl',
 'PuntNonReturn_elapsed_time_model.pkl',
 'PuntPenalty_model.pkl',
 'PuntPenaltyType_model.pkl',
 'PuntPosteamPostKickPenaltyType_model.pkl',
 'PuntPosteamPreKickPenalty_elapsed_time_model.pkl',
 'PuntPosteamPreKickPenaltyType_model.pkl',
 'PuntResult_model.pkl',
 'PuntReturn_elapsed_time_model.pkl',
 'PuntReturnDefFumbleTD_elapsed_time_model.pkl',
 'PuntReturnDefFumbleTD_model.pkl',
 'PuntReturnDefFumbleYards_model.pkl',
 'PuntReturnFumbleLost_elapsed_time_model.pkl',
 'PuntReturnFumbleLost_model.pkl',
 'PuntReturnYards_model.pkl']
for model in punt_models:
    model_var =model.split('_model.pkl')[0]
    exec(f"{model_var} = pickle.load(open(f'{model_dir}/PuntModels/{model}','rb'))")

rush_models=['NN_QBRedzoneRushingModel.pkl',
 'NN_QBRushElapsedTime.pkl',
 'NN_QBRushingModel.pkl',
 'NN_QBRushOB.pkl',
 'NN_RBRedzoneRushingModel.pkl',
 'NN_RBRushElapsedTime.pkl',
 'NN_RBRushingModel.pkl',
 'NN_RBRushOB.pkl',
 'NN_RushElapsedTime.pkl',
 'NN_RushFumbleElapsedTime.pkl',
 'NN_RushFumbleModel.pkl',
 'NN_RushFumbleReturnYardsModel.pkl',
 'NN_RushOB.pkl',
 'NN_WRRedzoneRushingModel.pkl',
 'NN_WRRushElapsedTime.pkl',
 'NN_WRRushingModel.pkl',
 'NN_WRRushOB.pkl',
 'NN_FourthQtrQBKneelElapsedTime.pkl',
 'NN_FourthQtrQBRushElapsedTime.pkl',
 'NN_FourthQtrRBRushElapsedTime.pkl',
 'NN_FourthQtrWRRushElapsedTime.pkl',]
# Rush Models
for model in rush_models:
    model_var = model.split('NN_')[1]
    model_var =model_var.split('.pkl')[0]
    exec(f"{model_var} = pickle.load(open(f'{model_dir}/RushingModels/{model}','rb'))")

RushModels={'QB':[QBRushingModel, QBRushElapsedTime, QBRushOB],
            'RB':[RBRushingModel, RBRushElapsedTime, RBRushOB],
            'WR':[WRRushingModel, WRRushElapsedTime, WRRushOB],}

RushModels4thQtr = {'QB':[QBRushingModel, FourthQtrQBRushElapsedTime, QBRushOB],
                    'RB':[RBRushingModel, FourthQtrRBRushElapsedTime, RBRushOB],
                    'WR':[WRRushingModel, FourthQtrWRRushElapsedTime, WRRushOB],}
# Pass Models
pass_models=['NN_AirYardsModel.pkl',
 'NN_CatchFumbleElapsedTime.pkl',
 'NN_CatchFumbleModel.pkl',
 'NN_CatchFumbleReturnYardsModel.pkl',
 'NN_deepPassModel.pkl',
 'NN_InterceptionReturnYardsModel.pkl',
 'NN_mediumPassModel.pkl',
 'NN_CompletePassElapsedTime.pkl',
 'NN_IncompletePassElapsedTime.pkl',
 'NN_passingModel.pkl',
 'NN_PassIntElapsedTime.pkl',
 'NN_RecOB.pkl',
 'NN_sackElapsedTimeModel.pkl',
 'NN_sackModel.pkl',
 'NN_sackYardsModel.pkl',
 'NN_shortPassModel.pkl',
 'NN_stripSackElapsedTimeModel.pkl',
 'NN_stripSackModel.pkl',
 'NN_stripSackReturnTDModel.pkl',
 'NN_stripSackReturnYardsModel.pkl',
 'NN_YACModel.pkl']
for model in pass_models:
    model_var = model.split('NN_')[1]
    model_var =model_var.split('.pkl')[0]
    exec(f"{model_var} = pickle.load(open(f'{model_dir}/ReceivingModels/{model}','rb'))")
    
# Timeout Models
timeout_models=['FirstHalfDefteam2MinTimeout_model.pkl',
 'FirstHalfDefteamTimeout_model.pkl',
 'FirstHalfPosteam2MinTimeout_model.pkl',
 'FirstHalfPosteamTimeout_model.pkl',
 'SecondHalfDefteam2MinTimeout_model.pkl',
 'SecondHalfDefteamTimeout_model.pkl',
 'SecondHalfPosteam2MinTimeout_model.pkl',
 'SecondHalfPosteamTimeout_model.pkl']
for model in timeout_models:
    model_var =model.split('_model.pkl')[0]
    exec(f"{model_var} = pickle.load(open(f'{model_dir}/TimeoutModels/{model}','rb'))")

# Penalty Models
penalty_models=[
 'complete_pass_penalty_defteam_et_model.pkl',
 'complete_pass_penalty_defteam_model.pkl',
 'complete_pass_penalty_model.pkl',
 'complete_pass_penalty_posteam_et_model.pkl',
 'complete_pass_penalty_posteam_model.pkl',
 'complete_pass_penalty_team_model.pkl',
 'incomplete_pass_penalty_defteam_et_model.pkl',
 'incomplete_pass_penalty_defteam_model.pkl',
 'incomplete_pass_penalty_model.pkl',
 'incomplete_pass_penalty_posteam_et_model.pkl',
 'incomplete_pass_penalty_posteam_model.pkl',
 'incomplete_pass_penalty_team_model.pkl',
 'pre_snap_penalty_defteam_et_model.pkl',
 'pre_snap_penalty_defteam_model.pkl',
 'pre_snap_penalty_model.pkl',
 'pre_snap_penalty_posteam_et_model.pkl',
 'pre_snap_penalty_posteam_model.pkl',
 'pre_snap_penalty_team_model.pkl',
 'rushing_penalty_defteam_et_model.pkl',
 'rushing_penalty_defteam_model.pkl',
 'rushing_penalty_model.pkl',
 'rushing_penalty_posteam_et_model.pkl',
 'rushing_penalty_posteam_model.pkl',
 'rushing_penalty_team_model.pkl']
for model in penalty_models:
    model_var =model.split('_model.pkl')[0]
    exec(f"{model_var} = pickle.load(open(f'{model_dir}/PenaltyModels/{model}','rb'))")
