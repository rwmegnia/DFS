#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 03:41:53 2022

@author: robertmegnia
"""

# Configure Dictionaries for database updates
PBP_STATIC_COLUMNS={'gsis_id':'gsis_id',
                    'week':'week',
                    'game_date':'game_date',
                    'start_time':'start_time',
                    'total_line':'total_line',
                    'proj_posteam_score':'proj_team_score',
                    'posteam':'team',
                    'defteam':'opp',
                    'posteam_type':'game_location',
                    'game_id':'game_id'}

PBP_DST_STATIC_COLUMNS={'gsis_id':'gsis_id',
                         'week':'week',
                         'home_score':'home_score',
                         'away_score':'away_score',
                         'game_date':'game_date',
                         'start_time':'start_time',
                         'total_line':'total_line',
                         'proj_defteam_score':'proj_team_score',
                         'defteam':'full_name',
                         'posteam':'opp',
                         'posteam_type':'game_location',
                         'game_id':'game_id'}


PBP_PASS_STATS_COLUMNS={'passing_yards':'pass_yards',
                        'air_yards':'pass_air_yards',
                        'pass_touchdown':'pass_td',
                        'pass_attempt':'pass_att',
                        'complete_pass':'pass_cmp',
                        'fumble_lost':'pass_fumble_lost',
                        'interception':'int',
                        'sack':'sacks',
                        'qb_hit':'qb_hit',
                        'two_point_conv_result':'pass_two_point_conv'}

PBP_RUSH_STATS_COLUMNS={'rushing_yards':'rush_yards',
                        'rush_touchdown':'rush_td',
                        'rush_attempt':'rush_att',
                        'fumble_lost':'rush_fumble_lost',
                        'tackled_for_loss':'tfl',
                        'two_point_conv_result':'rush_two_point_conv'}

PBP_REC_STATS_COLUMNS={'receiving_yards':'rec_yards',
                       'air_yards':'rec_air_yards',
                       'yards_after_catch':'yac',
                       'pass_touchdown':'rec_td',
                       'pass_attempt':'targets',
                       'complete_pass':'rec',
                       'fumble_lost':'rec_fumble_lost',
                       'two_point_conv_result':'rec_two_point_conv'}

PBP_DST_STATS_COLUMNS={'fumble_lost':'fumble_recoveries',
                       'sack':'sack',
                       'interception':'interception',
                       'punt_blocked':'punt_blocked',
                       'fg_blocked':'fg_blocked',
                       'safety':'safety',
                       'return_touchdown':'return_touchdown',
                       'int_return_td':'int_return_td',
                       'fumble_return_td':'fumble_return_td',
                       'fg_return_td':'fg_return_td',
                       'fg_block_return_td':'fg_block_return_td',
                       'punt_return_td':'punt_return_td',
                       'kick_return_td':'kick_return_td',
                       'punt_block_return_td':'punt_block_return_td',
                       'tackled_for_loss':'tfl',
                       'qb_hit':'qb_hit'}