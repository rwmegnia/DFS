#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 05:43:13 2022

@author: robertmegnia

Column Mappings for Model Features
"""

# Machine Learning Columns
# Non- Features
ML_OffenseNonFeatures = [
    "full_name",
    "gsis_id",
    "opp",
    "injury_status",
    "rookie_year",
    "height",
    "weight",
    "draft_number",
    "draft_round",
    "college",
    "college_conference",
    "player_id",
    "week",
    "position",
    "season",
    "game_date",
    "start_time",
    "team",
    "game_location",
    "game_id",
    "game_day",
    "Slate",
    "poe",
    "exDKPts",
    "Stochastic",
    "Median",
    "Floor",
    "Ceiling",
    "UpsideProb",
    "UpsideScore",
    "RosterPercent",
]
ML_DefenseNonFeatures = [
    "full_name",
    "gsis_id",
    "opp",
    "week",
    "position",
    "season",
    "game_date",
    "start_time",
    "team",
    "game_location",
    "game_id",
    "game_day",
    "Slate",
    "Stochastic",
    "Median",
    "Floor",
    "Ceiling",
    "UpsideProb",
    "UpsideScore",
    "RosterPercent",
]
# ML Opponent Features Columns
DK_Stoch_stats = [
    "pass_yards",
    "pass_td",
    "rush_yards",
    "rush_td",
    "rec",
    "rec_yards",
    "rec_td",
    "fumbles_lost",
    "int",
    "DKPts",
]

DK_DST_Stoch_stats = [
    "fumble_recoveries",
    "interception",
    "sack",
    "blocks",
    "safety",
    "return_touchdown",
    "points_allowed",
    "DKPts",
]
# ML Known Features
ML_KnownFeatures = [
    "total_line",
    "proj_team_score",
    "spread_line",
    "opp_Rank",
    "Adj_opp_Rank",
    "salary",
    "depth_team",
    "matchup",
]
# ML Team Share Features
MLTeamShareFeatures = [
    "DKPts_share_pos",
    "DKPts_share_skill_pos",
    "DKPts_share",
    "rushing_DKPts_share",
    "receiving_DKPts_share",
    "pass_yards_share",
    "pass_td_share",
    "rush_yards_share",
    "rush_td_share",
    "rec_yards_share",
    "rec_td_share",
    "rec_share",
    "fumbles_lost_share",
    "int_share",
]
# ML Position Model Features
ML_OFFENSE_FEATURE_COLS=[
    'pass_yards',
     'pass_air_yards',
     'pass_td',
     'pass_att',
     'pass_cmp',
     'pass_fumble_lost',
     'int',
     'sacks',
     'qb_hit',
     'rush_yards',
     'rush_td',
     'rush_att',
     'rush_fumble_lost',
     'tfl',
     'rush_share',
     'rush_redzone_looks',
     'rush_value',
     'rec_yards',
     'rec_air_yards',
     'yac',
     'rec_td',
     'targets',
     'rec',
     'rec_fumble_lost',
     'target_share',
     'air_yards_share',
     'rec_redzone_looks',
     'wopr',
     'target_value',
     'offense_epa',
     'offense_pass_epa',
     'offense_rush_epa',
     'defense_epa',
     'defense_pass_epa',
     'defense_rush_epa',
     'fumbles_lost',
     'Usage',
     'HVU',
     'pass_yds_per_att',
     'rush_yds_per_att',
     'passer_rating',
     'adot',
     'passing_DKPts',
     'rushing_DKPts',
     'receiving_DKPts',
     'avg_DKPts',
     'PPO',
     'HV_PPO',
     'team_DKPts_pos',
     'DKPts_share_pos',
     'team_DKPts_skill_pos',
     'DKPts_share_skill_pos',
     'team_DKPts',
     'DKPts_share',
     'team_pass_yards',
     'pass_yards_share',
     'team_pass_td',
     'pass_td_share',
     'team_rush_yards',
     'rush_yards_share',
     'team_rush_td',
     'rush_td_share',
     'team_rec_yards',
     'rec_yards_share',
     'team_rec_td',
     'rec_td_share',
     'team_rec',
     'rec_share',
     'team_fumbles_lost',
     'fumbles_lost_share',
     'team_int',
     'int_share',
     'team_passing_DKPts',
     'passing_DKPts_share',
     'team_rushing_DKPts',
     'rushing_DKPts_share',
     'team_receiving_DKPts',
     'receiving_DKPts_share',
     'Rank',
     'offensive_snapcounts',
     'offensive_snapcount_percentage',
     'total_line',
     'proj_team_score',
     'spread_line',
     'opp_Rank',
     'Adj_opp_Rank',
     'salary',
     'depth_team',
     'matchup',
     'pass_yards_allowed',
     'pass_air_yards_allowed',
     'pass_td_allowed',
     'pass_att_allowed',
     'pass_cmp_allowed',
     'pass_fumble_lost_allowed',
     'int_allowed',
     'sacks_allowed',
     'qb_hit_allowed',
     'rush_yards_allowed',
     'rush_td_allowed',
     'rush_att_allowed',
     'rush_fumble_lost_allowed',
     'tfl_allowed',
     'rush_share_allowed',
     'rush_redzone_looks_allowed',
     'rush_value_allowed',
     'rec_yards_allowed',
     'rec_air_yards_allowed',
     'yac_allowed',
     'rec_td_allowed',
     'targets_allowed',
     'rec_allowed',
     'rec_fumble_lost_allowed',
     'target_share_allowed',
     'air_yards_share_allowed',
     'rec_redzone_looks_allowed',
     'wopr_allowed',
     'target_value_allowed',
     'offense_epa_allowed',
     'offense_pass_epa_allowed',
     'offense_rush_epa_allowed',
     'defense_epa_allowed',
     'defense_pass_epa_allowed',
     'defense_rush_epa_allowed',
     'fumbles_lost_allowed',
     'Usage_allowed',
     'HVU_allowed',
     'pass_yds_per_att_allowed',
     'rush_yds_per_att_allowed',
     'passer_rating_allowed',
     'adot_allowed',
     'passing_DKPts_allowed',
     'rushing_DKPts_allowed',
     'receiving_DKPts_allowed',
     'DKPts_allowed',
     'PPO_allowed',
     'HV_PPO_allowed',
     'team_DKPts_pos_allowed',
     'team_DKPts_skill_pos_allowed',
     'team_DKPts_allowed',
     'team_pass_yards_allowed',
     'team_pass_td_allowed',
     'team_rush_yards_allowed',
     'team_rush_td_allowed',
     'team_rec_yards_allowed',
     'team_rec_td_allowed',
     'team_rec_allowed',
     'team_fumbles_lost_allowed',
     'team_int_allowed',
     'team_passing_DKPts_allowed',
     'passing_DKPts_share_allowed',
     'team_rushing_DKPts_allowed',
     'team_receiving_DKPts_allowed',
     'Rank_allowed',
     'offensive_snapcounts_allowed',
     'offensive_snapcount_percentage_allowed']

ML_DEFENSE_FEATURE_COLS=['fumble_recoveries',
 'sack',
 'qb_hit',
 'tfl',
 'interception',
 'blocks',
 'safety',
 'return_touchdown',
 'points_allowed',
 'offense_epa',
 'offense_pass_epa',
 'offense_rush_epa',
 'defense_epa',
 'defense_pass_epa',
 'defense_rush_epa',
 'avg_DKPts',
 'Rank',
 'DepthChart',
 'total_line',
 'proj_team_score',
 'spread_line',
 'opp_Rank',
 'Adj_opp_Rank',
 'salary',
 'depth_team',
 'fumble_recoveries_allowed',
 'sack_allowed',
 'qb_hit_allowed',
 'tfl_allowed',
 'interception_allowed',
 'blocks_allowed',
 'safety_allowed',
 'return_touchdown_allowed',
 'points_allowed_allowed',
 'offense_epa_allowed',
 'offense_pass_epa_allowed',
 'offense_rush_epa_allowed',
 'defense_epa_allowed',
 'defense_pass_epa_allowed',
 'defense_rush_epa_allowed',
 'DKPts_allowed',
 'Rank_allowed',
 'DepthChart_allowed']

###### ROOKIE MODEL COLUMNS MAPPINGS

RookieNonFeatures = [
        "full_name",
        "gsis_id",
        "opp",
        "week",
        "position",
        "season",
        "rookie_year",
        "game_date",
        "team",
        "game_location",
        "game_day",
        "Slate",
    ]

RookieKnownFeatures=['total_line',
               'proj_team_score',
               'spread_line',
               'opp_Rank',
               'Adj_opp_Rank',
               'depth_team',
               'salary']

# Rookie Model Features

QBRookieFeatures=['team_DKPts',
                'team_DKPts_allowed',
                'team_passing_DKPts',
                'team_passing_DKPts_allowed']


RBRookieFeatures=['team_DKPts',
                  'team_DKPts_allowed',
                  'team_rushing_DKPts',
                  'team_rushing_DKPts_allowed',
                  'team_receiving_DKPts',
                  'team_receiving_DKPts_allowed']

RecRookieFeatures=['team_DKPts',
                  'team_DKPts_allowed',
                  'team_passing_DKPts',
                  'team_passing_DKPts_allowed',
                  'team_receiving_DKPts',
                  'team_receiving_DKPts_allowed']

RookieFeatures={
    'QB':QBRookieFeatures,
    'RB':RBRookieFeatures,
    'WR':RecRookieFeatures,
    'TE':RecRookieFeatures}

# Player Shares Model Columns Mappings
PSM_NonFeatures=['full_name',
             'gsis_id',
             'opp',
             'week',
             'position',
             'season',
             'game_date',
             'team',
             'game_location',
             'game_day',
             'Slate',]

PSM_KnownFeatures=['total_line',
               'proj_team_score',
               'spread_line',
               'opp_Rank',
               'Adj_opp_Rank',
               'depth_team']

# Rush Share features
rush_DKPts_share_features=[
    'rush_redzone_looks',
    'rush_yards',
    'rushing_DKPts',
    'rush_att',
    'rush_share',
    'rushing_DKPts_share',
    'depth_team']

# Rush Share features
receiving_DKPts_share_features=[
    'rec_redzone_looks',
    'rec_yards',
    'receiving_DKPts',
    'targets',
    'target_share',
    'rec_share',
    'receiving_DKPts_share',
    'depth_team']

PlayerShareModel_features_dict={
    'receiving_DKPts_share':receiving_DKPts_share_features,
    'rushing_DKPts_share':rush_DKPts_share_features}

# Team Depth Chart Models
TDM_NonFeatures=['team',
             'week',
             'season',
             'game_date',
             'game_id',
             'Location',
             'opp',
             'QB1_full_name',
             'RB1_full_name',
             'RB2_full_name',
             'RB3_full_name',
             'WR1_full_name',
             'WR2_full_name',
             'WR3_full_name',
             'WR4_full_name',
             'WR5_full_name',
             'TE1_full_name',
             'TE2_full_name',
             'TE3_full_name',]

TDM_KnownFeatures=['total_line',
               'ImpliedPoints',
               'spread_line',
               'QB1_salary',
               'RB1_salary',
               'RB2_salary',
               'RB3_salary',
               'WR1_salary',
               'WR2_salary',
               'WR3_salary',
               'WR4_salary',
               'WR5_salary',
               'TE1_salary',
               'TE2_salary',
               'TE3_salary']

feature_cols=['air_epa',
 'air_yards',
 'comp_air_epa',
 'comp_yac_epa',
 'complete_pass',
 'epa',
 'fumble',
 'fumble_lost',
 'interception',
 'pass_attempt',
 'passing_yards',
 'pass_touchdown',
 'qb_epa',
 'qb_hit',
 'sack',
 'yards_after_catch',
 'big_pass_play',
 'rush_attempt',
 'rush_touchdown',
 'rushing_yards',
 'tackled_for_loss',
 'big_run',
 'early_down_pass_rate',
 'early_down_success_rate',
 '3rdDown_success_rate',
 '4thDown_attempt_rate',
 '4thDown_success_rate',
 'EarlyGame_4thDown_attempt_rate',
 'EarlyGame_4thDown_success_rate',
 'Points',
 'TotalPoints',
 'OppPoints',
 'ADOT',
 'passer_rating',
 'ypa',
 'QB1_pass_cmp',
 'QB1_pass_att',
 'QB1_pass_yards',
 'QB1_pass_yds_per_att',
 'QB1_pass_air_yards',
 'QB1_pass_td',
 'QB1_int',
 'QB1_DKPts',
 'QB1_DKPts_share_pos',
 'QB1_passing_DKPts_share',
 'QB1_rushing_DKPts_share',
 'QB1_receiving_DKPts_share',
 'QB1_passer_rating',
 'RB1_offensive_snapcounts',
 'RB1_rush_att',
 'RB1_rush_share',
 'RB1_rush_yards',
 'RB1_rush_td',
 'RB1_targets',
 'RB1_rec',
 'RB1_rec_yards',
 'RB1_rec_td',
 'RB1_rush_redzone_looks',
 'RB1_PPO',
 'RB1_DKPts',
 'RB1_DKPts_share_pos',
 'RB1_passing_DKPts_share',
 'RB1_rushing_DKPts_share',
 'RB1_receiving_DKPts_share',
 'RB2_offensive_snapcounts',
 'RB2_rush_att',
 'RB2_rush_share',
 'RB2_rush_yards',
 'RB2_rush_td',
 'RB2_targets',
 'RB2_rec',
 'RB2_rec_yards',
 'RB2_rec_td',
 'RB2_rush_redzone_looks',
 'RB2_PPO',
 'RB2_DKPts',
 'RB2_DKPts_share_pos',
 'RB2_passing_DKPts_share',
 'RB2_rushing_DKPts_share',
 'RB2_receiving_DKPts_share',
 'RB3_offensive_snapcounts',
 'RB3_rush_att',
 'RB3_rush_share',
 'RB3_rush_yards',
 'RB3_rush_td',
 'RB3_targets',
 'RB3_rec',
 'RB3_rec_yards',
 'RB3_rec_td',
 'RB3_rush_redzone_looks',
 'RB3_PPO',
 'RB3_DKPts',
 'RB3_DKPts_share_pos',
 'RB3_passing_DKPts_share',
 'RB3_rushing_DKPts_share',
 'RB3_receiving_DKPts_share',
 'WR1_offensive_snapcounts',
 'WR1_targets',
 'WR1_rec',
 'WR1_rec_yards',
 'WR1_yac',
 'WR1_rec_air_yards',
 'WR1_wopr',
 'WR1_rec_td',
 'WR1_rec_redzone_looks',
 'WR1_PPO',
 'WR1_DKPts',
 'WR1_DKPts_share_pos',
 'WR1_passing_DKPts_share',
 'WR1_rushing_DKPts_share',
 'WR1_receiving_DKPts_share',
 'WR2_offensive_snapcounts',
 'WR2_targets',
 'WR2_rec',
 'WR2_rec_yards',
 'WR2_yac',
 'WR2_rec_air_yards',
 'WR2_wopr',
 'WR2_rec_td',
 'WR2_rec_redzone_looks',
 'WR2_PPO',
 'WR2_DKPts',
 'WR2_DKPts_share_pos',
 'WR2_passing_DKPts_share',
 'WR2_rushing_DKPts_share',
 'WR2_receiving_DKPts_share',
 'WR3_offensive_snapcounts',
 'WR3_targets',
 'WR3_rec',
 'WR3_rec_yards',
 'WR3_yac',
 'WR3_rec_air_yards',
 'WR3_wopr',
 'WR3_rec_td',
 'WR3_rec_redzone_looks',
 'WR3_PPO',
 'WR3_DKPts',
 'WR3_DKPts_share_pos',
 'WR3_passing_DKPts_share',
 'WR3_rushing_DKPts_share',
 'WR3_receiving_DKPts_share',
 'WR4_offensive_snapcounts',
 'WR4_targets',
 'WR4_rec',
 'WR4_rec_yards',
 'WR4_yac',
 'WR4_rec_air_yards',
 'WR4_wopr',
 'WR4_rec_td',
 'WR4_rec_redzone_looks',
 'WR4_PPO',
 'WR4_DKPts',
 'WR4_DKPts_share_pos',
 'WR4_passing_DKPts_share',
 'WR4_rushing_DKPts_share',
 'WR4_receiving_DKPts_share',
 'WR5_offensive_snapcounts',
 'WR5_targets',
 'WR5_rec',
 'WR5_rec_yards',
 'WR5_yac',
 'WR5_rec_air_yards',
 'WR5_wopr',
 'WR5_rec_td',
 'WR5_rec_redzone_looks',
 'WR5_PPO',
 'WR5_DKPts',
 'WR5_DKPts_share_pos',
 'WR5_passing_DKPts_share',
 'WR5_rushing_DKPts_share',
 'WR5_receiving_DKPts_share',
 'TE1_offensive_snapcounts',
 'TE1_targets',
 'TE1_rec',
 'TE1_rec_yards',
 'TE1_yac',
 'TE1_rec_air_yards',
 'TE1_wopr',
 'TE1_rec_td',
 'TE1_rec_redzone_looks',
 'TE1_PPO',
 'TE1_DKPts',
 'TE1_DKPts_share_pos',
 'TE1_passing_DKPts_share',
 'TE1_rushing_DKPts_share',
 'TE1_receiving_DKPts_share',
 'TE2_offensive_snapcounts',
 'TE2_targets',
 'TE2_rec',
 'TE2_rec_yards',
 'TE2_yac',
 'TE2_rec_air_yards',
 'TE2_wopr',
 'TE2_rec_td',
 'TE2_rec_redzone_looks',
 'TE2_PPO',
 'TE2_DKPts',
 'TE2_DKPts_share_pos',
 'TE2_passing_DKPts_share',
 'TE2_rushing_DKPts_share',
 'TE2_receiving_DKPts_share',
 'TE3_offensive_snapcounts',
 'TE3_targets',
 'TE3_rec',
 'TE3_rec_yards',
 'TE3_yac',
 'TE3_rec_air_yards',
 'TE3_wopr',
 'TE3_rec_td',
 'TE3_rec_redzone_looks',
 'TE3_PPO',
 'TE3_DKPts',
 'TE3_DKPts_share_pos',
 'TE3_passing_DKPts_share',
 'TE3_rushing_DKPts_share',
 'TE3_receiving_DKPts_share',
 'passing_fpts',
 'rushing_fpts',
 'receiving_fpts',
 'team_fpts',
 'total_line',
 'ImpliedPoints',
 'spread_line',
 'QB1_salary',
 'RB1_salary',
 'RB2_salary',
 'RB3_salary',
 'WR1_salary',
 'WR2_salary',
 'WR3_salary',
 'WR4_salary',
 'WR5_salary',
 'TE1_salary',
 'TE2_salary',
 'TE3_salary',
 'air_epa_allowed',
 'air_yards_allowed',
 'comp_air_epa_allowed',
 'comp_yac_epa_allowed',
 'complete_pass_allowed',
 'epa_allowed',
 'fumble_allowed',
 'fumble_lost_allowed',
 'interception_allowed',
 'pass_attempt_allowed',
 'passing_yards_allowed',
 'pass_touchdown_allowed',
 'qb_epa_allowed',
 'qb_hit_allowed',
 'sack_allowed',
 'yards_after_catch_allowed',
 'big_pass_play_allowed',
 'rush_attempt_allowed',
 'rush_touchdown_allowed',
 'rushing_yards_allowed',
 'tackled_for_loss_allowed',
 'big_run_allowed',
 'early_down_pass_rate_allowed',
 'early_down_success_rate_allowed',
 '3rdDown_success_rate_allowed',
 '4thDown_attempt_rate_allowed',
 '4thDown_success_rate_allowed',
 'EarlyGame_4thDown_attempt_rate_allowed',
 'EarlyGame_4thDown_success_rate_allowed',
 'Points_allowed',
 'TotalPoints_allowed',
 'OppPoints_allowed',
 'ADOT_allowed',
 'passer_rating_allowed',
 'ypa_allowed',
 'QB1_pass_cmp_allowed',
 'QB1_pass_att_allowed',
 'QB1_pass_yards_allowed',
 'QB1_pass_yds_per_att_allowed',
 'QB1_pass_air_yards_allowed',
 'QB1_pass_td_allowed',
 'QB1_int_allowed',
 'QB1_DKPts_allowed',
 'QB1_DKPts_share_pos_allowed',
 'QB1_passing_DKPts_share_allowed',
 'QB1_rushing_DKPts_share_allowed',
 'QB1_receiving_DKPts_share_allowed',
 'QB1_passer_rating_allowed',
 'RB1_offensive_snapcounts_allowed',
 'RB1_rush_att_allowed',
 'RB1_rush_share_allowed',
 'RB1_rush_yards_allowed',
 'RB1_rush_td_allowed',
 'RB1_targets_allowed',
 'RB1_rec_allowed',
 'RB1_rec_yards_allowed',
 'RB1_rec_td_allowed',
 'RB1_rush_redzone_looks_allowed',
 'RB1_PPO_allowed',
 'RB1_DKPts_allowed',
 'RB1_DKPts_share_pos_allowed',
 'RB1_passing_DKPts_share_allowed',
 'RB1_rushing_DKPts_share_allowed',
 'RB1_receiving_DKPts_share_allowed',
 'RB2_offensive_snapcounts_allowed',
 'RB2_rush_att_allowed',
 'RB2_rush_share_allowed',
 'RB2_rush_yards_allowed',
 'RB2_rush_td_allowed',
 'RB2_targets_allowed',
 'RB2_rec_allowed',
 'RB2_rec_yards_allowed',
 'RB2_rec_td_allowed',
 'RB2_rush_redzone_looks_allowed',
 'RB2_PPO_allowed',
 'RB2_DKPts_allowed',
 'RB2_DKPts_share_pos_allowed',
 'RB2_passing_DKPts_share_allowed',
 'RB2_rushing_DKPts_share_allowed',
 'RB2_receiving_DKPts_share_allowed',
 'RB3_offensive_snapcounts_allowed',
 'RB3_rush_att_allowed',
 'RB3_rush_share_allowed',
 'RB3_rush_yards_allowed',
 'RB3_rush_td_allowed',
 'RB3_targets_allowed',
 'RB3_rec_allowed',
 'RB3_rec_yards_allowed',
 'RB3_rec_td_allowed',
 'RB3_rush_redzone_looks_allowed',
 'RB3_PPO_allowed',
 'RB3_DKPts_allowed',
 'RB3_DKPts_share_pos_allowed',
 'RB3_passing_DKPts_share_allowed',
 'RB3_rushing_DKPts_share_allowed',
 'RB3_receiving_DKPts_share_allowed',
 'WR1_offensive_snapcounts_allowed',
 'WR1_targets_allowed',
 'WR1_rec_allowed',
 'WR1_rec_yards_allowed',
 'WR1_yac_allowed',
 'WR1_rec_air_yards_allowed',
 'WR1_wopr_allowed',
 'WR1_rec_td_allowed',
 'WR1_rec_redzone_looks_allowed',
 'WR1_PPO_allowed',
 'WR1_DKPts_allowed',
 'WR1_DKPts_share_pos_allowed',
 'WR1_passing_DKPts_share_allowed',
 'WR1_rushing_DKPts_share_allowed',
 'WR1_receiving_DKPts_share_allowed',
 'WR2_offensive_snapcounts_allowed',
 'WR2_targets_allowed',
 'WR2_rec_allowed',
 'WR2_rec_yards_allowed',
 'WR2_yac_allowed',
 'WR2_rec_air_yards_allowed',
 'WR2_wopr_allowed',
 'WR2_rec_td_allowed',
 'WR2_rec_redzone_looks_allowed',
 'WR2_PPO_allowed',
 'WR2_DKPts_allowed',
 'WR2_DKPts_share_pos_allowed',
 'WR2_passing_DKPts_share_allowed',
 'WR2_rushing_DKPts_share_allowed',
 'WR2_receiving_DKPts_share_allowed',
 'WR3_offensive_snapcounts_allowed',
 'WR3_targets_allowed',
 'WR3_rec_allowed',
 'WR3_rec_yards_allowed',
 'WR3_yac_allowed',
 'WR3_rec_air_yards_allowed',
 'WR3_wopr_allowed',
 'WR3_rec_td_allowed',
 'WR3_rec_redzone_looks_allowed',
 'WR3_PPO_allowed',
 'WR3_DKPts_allowed',
 'WR3_DKPts_share_pos_allowed',
 'WR3_passing_DKPts_share_allowed',
 'WR3_rushing_DKPts_share_allowed',
 'WR3_receiving_DKPts_share_allowed',
 'WR4_offensive_snapcounts_allowed',
 'WR4_targets_allowed',
 'WR4_rec_allowed',
 'WR4_rec_yards_allowed',
 'WR4_yac_allowed',
 'WR4_rec_air_yards_allowed',
 'WR4_wopr_allowed',
 'WR4_rec_td_allowed',
 'WR4_rec_redzone_looks_allowed',
 'WR4_PPO_allowed',
 'WR4_DKPts_allowed',
 'WR4_DKPts_share_pos_allowed',
 'WR4_passing_DKPts_share_allowed',
 'WR4_rushing_DKPts_share_allowed',
 'WR4_receiving_DKPts_share_allowed',
 'WR5_offensive_snapcounts_allowed',
 'WR5_targets_allowed',
 'WR5_rec_allowed',
 'WR5_rec_yards_allowed',
 'WR5_yac_allowed',
 'WR5_rec_air_yards_allowed',
 'WR5_wopr_allowed',
 'WR5_rec_td_allowed',
 'WR5_rec_redzone_looks_allowed',
 'WR5_PPO_allowed',
 'WR5_DKPts_allowed',
 'WR5_DKPts_share_pos_allowed',
 'WR5_passing_DKPts_share_allowed',
 'WR5_rushing_DKPts_share_allowed',
 'WR5_receiving_DKPts_share_allowed',
 'TE1_offensive_snapcounts_allowed',
 'TE1_targets_allowed',
 'TE1_rec_allowed',
 'TE1_rec_yards_allowed',
 'TE1_yac_allowed',
 'TE1_rec_air_yards_allowed',
 'TE1_wopr_allowed',
 'TE1_rec_td_allowed',
 'TE1_rec_redzone_looks_allowed',
 'TE1_PPO_allowed',
 'TE1_DKPts_allowed',
 'TE1_DKPts_share_pos_allowed',
 'TE1_passing_DKPts_share_allowed',
 'TE1_rushing_DKPts_share_allowed',
 'TE1_receiving_DKPts_share_allowed',
 'TE2_offensive_snapcounts_allowed',
 'TE2_targets_allowed',
 'TE2_rec_allowed',
 'TE2_rec_yards_allowed',
 'TE2_yac_allowed',
 'TE2_rec_air_yards_allowed',
 'TE2_wopr_allowed',
 'TE2_rec_td_allowed',
 'TE2_rec_redzone_looks_allowed',
 'TE2_PPO_allowed',
 'TE2_DKPts_allowed',
 'TE2_DKPts_share_pos_allowed',
 'TE2_passing_DKPts_share_allowed',
 'TE2_rushing_DKPts_share_allowed',
 'TE2_receiving_DKPts_share_allowed',
 'TE3_offensive_snapcounts_allowed',
 'TE3_targets_allowed',
 'TE3_rec_allowed',
 'TE3_rec_yards_allowed',
 'TE3_yac_allowed',
 'TE3_rec_air_yards_allowed',
 'TE3_wopr_allowed',
 'TE3_rec_td_allowed',
 'TE3_rec_redzone_looks_allowed',
 'TE3_PPO_allowed',
 'TE3_DKPts_allowed',
 'TE3_DKPts_share_pos_allowed',
 'TE3_passing_DKPts_share_allowed',
 'TE3_rushing_DKPts_share_allowed',
 'TE3_receiving_DKPts_share_allowed',
 'passing_fpts_allowed',
 'rushing_fpts_allowed',
 'receiving_fpts_allowed',
 'team_fpts_allowed']

# Team Stats Models

team_fpts_features=[
    'ImpliedPoints',
    'TotalPoints',
    'spread_line',
    'team_fpts',
    'team_fpts_allowed']

# 
team_pass_attempt_features=[
                            'ImpliedPoints',
                            'TotalPoints',
                            'spread_line',
                            'team_fpts',
                            'pass_attempt',
                            'pass_attempt_allowed',
                            'rush_attempt',
                            'rush_attempt_allowed']

# 
team_rush_attempt_features=[
                            'ImpliedPoints',
                            'TotalPoints',
                            'spread_line',
                            'team_fpts',
                            'pass_attempt',
                            'pass_attempt_allowed',
                            'rush_attempt',
                            'rush_attempt_allowed',
                            'epa']

# Features to predict passing fantasy points
team_passing_DKPts_features=[
    'ImpliedPoints',
    'TotalPoints',
    'spread_line',
    'team_fpts',
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',]

# Features to predict rushing fantasy points
team_rushing_DKPts_features=[
    'ImpliedPoints',
    'TotalPoints',
    'spread_line',
    'team_fpts',
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',
    'rushing_fpts',
    'rushing_fpts_allowed',]

# Features to predict rushing fantasy points
team_receiving_DKPts_features=[
    'ImpliedPoints',
    'TotalPoints',
    'spread_line',
    'team_fpts',
    'pass_attempt',
    'rush_attempt',
    'passing_fpts',
    'passing_fpts_allowed',
    'rushing_fpts',
    'rushing_fpts_allowed',
    'receiving_fpts',
    'receiving_fpts_allowed']


# Columns that we know the values of going into a week
TSM_KnownFeatures=[
               'ImpliedPoints',
               'spread_line',
               'TotalPoints',]


TSM_NonFeatures=['team',
             'home_team',
             'away_team',
             'home_score',
             'away_score',
             'proj_away_score',
             'proj_home_score',
             'week',
             'season',
             'game_date',
             'game_id',
             'Location',
             'opp',
             'QB1_full_name',
             'RB1_full_name',
             'RB2_full_name',
             'RB3_full_name',
             'WR1_full_name',
             'WR2_full_name',
             'WR3_full_name',
             'WR4_full_name',
             'WR5_full_name',
             'TE1_full_name',
             'TE2_full_name',
             'TE3_full_name',
             'QB1_opp',
             'RB1_opp',
             'RB2_opp',
             'RB3_opp',
             'WR1_opp',
             'WR2_opp',
             'WR3_opp',
             'WR4_opp',
             'WR5_opp',
             'TE1_opp',
             'TE2_opp',
             'TE3_opp',]

TeamStatsModel_features_dict={
    'team_fpts':team_fpts_features,
    'pass_attempt':team_pass_attempt_features,
    'rush_attempt':team_rush_attempt_features,
    'passing_fpts':team_passing_DKPts_features,
    'rushing_fpts':team_rushing_DKPts_features,
    'receiving_fpts':team_receiving_DKPts_features}
