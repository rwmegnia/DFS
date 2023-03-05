#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:24:48 2022

@author: robertmegnia
"""

import pandas as pd
import numpy as np
from SimulationModels import *
from messageHandler import messageHandler
import time
import traceback

import sys
import os
from Team import Team
from Utils import *
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir = f"{basedir}/ml_models"
coaches = pd.read_csv("./coaches/coaching_logs.csv")
#%%
'''
Work to be done

* QB Hit
* QB Scramble
* Block Field Goals
* Redzone Models
* Penalties

'''


class Game:
    def __init__(self, home_team, away_team,iteration):
        self.iteration=iteration
        self.home_team = home_team
        self.away_team = away_team
        self.home_timeouts_remaining = 3
        self.away_timeouts_remaining = 3
        self.posteam_timeouts_remaining=3
        self.defteam_timeouts_remaining=3
        self.posteam_timeout = FirstHalfPosteamTimeout
        self.defteam_timeout = FirstHalfDefteamTimeout
        self.home_team_score = 0
        self.away_team_score = 0
        self.posteam_score = 0
        self.away_team_score = 0
        self.qtr = 1
        self.half = 1
        self.game_seconds_remaining = 3600
        self.half_seconds_remaining = 1800
        self.qtr_seconds_remaining = 900
        self.play_clock = 40
        self.pass_et = []
        self.end_game = False
        self.overTime = False
        self.OT_first_possession = True
        self.messageHandler = messageHandler
        self.export_stats = exportStats
        self.stats=stats=pd.DataFrame({
                                        'rush_attempt':[],
                                        'pass_attempt':[],
                                        'complete_pass':[],
                                        'rusher_player_id':[],
                                        'passer_player_id':[],
                                        'receiver_player_id':[],
                                        'passing_yards':[],
                                        'rushing_yards':[],
                                        'pass_touchdown':[],
                                        'rush_touchdown':[],
                                        'interception':[],
                                        'sack':[],
                                        'defteam':[],
                                        'posteam':[],
                                        })
        self.posteam, self.defteam = self.coinToss()
        self.kickoff()


    def coinToss(self):
        guess = np.random.choice(["heads", "tails"])
        result = np.random.choice(["heads", "tails"])
        decision = np.random.choice(["kick", "receive"])
        if self.overTime == True:
            decision = "receive"
        if guess == result:
            if decision == "kick":
                self.second_half_receiving_team = self.away_team
                self.posteam_score = self.home_team_score
                self.defteam_score = self.away_team_score
            else:
                self.second_half_receiving_team = self.home_team
                self.posteam_score = self.away_team_score
                self.defteam_score = self.home_team_score
        else:
            if decision == "kick":
                self.second_half_receiving_team = self.home_team
                self.second_half_kicking_team = self.away_team
                self.posteam_score = self.away_team_score
                self.defteam_score = self.home_team_score
            else:
                self.second_half_receiving_team = self.away_team
                self.second_half_kicking_team = self.home_team
                self.posteam_score = self.home_team_score
                self.defteam_score = self.away_team_score
        self.messageHandler(self,'Opening Coin Toss',
                            self.away_team,
                            self.home_team,
                            guess,
                            result,
                            decision)
        return self.home_team, self.away_team


    def kickoff(self):
        if self.end_game==True:
            return
        if self.half_seconds_remaining == 1800:
            kickoff_from = 65
        else:
            kickoff_from = 100 - KickoffFrom_model.predict([self.qtr])[0]
        if (self.half_seconds_remaining == 1800) & (self.half == 2):
            if self.posteam != self.second_half_receiving_team:
                self.change_possession()
        self.yardline = kickoff_from
        kickoff_result = KickoffResult_model.predict([self.qtr])[0]
        print(
            f"{self.defteam.team_abbrev} kicks from their own {kickoff_from} yardline"
        )
        if kickoff_result == "out_of_bounds":
            print("kickoff is out of bounds")
            self.yardline = 40
            self.start_drive()
        elif kickoff_result == "touchback":
            self.yardline = TouchbackEndYardLine_model.predict([self.qtr])[0]
            if self.yardline < 25:
                penalty_yards = 25 - self.yardline
                print(
                    f"Kickoff ends with a touchback. Penalty on {self.posteam.team_abbrev} for {penalty_yards} yards"
                )
                print(
                    f"{self.posteam.team_abbrev} starts drive at {self.yardline}"
                )
            elif self.yardline > 25:
                penalty_yards = self.yardline - 25
                print(
                    f"Kickoff ends with a touchback. Penalty on {self.defteam.team_abbrev} for {penalty_yards} yards"
                )
                print(
                    f"{self.posteam.team_abbrev} starts drive at {self.yardline}"
                )
            else:
                print(f"Kickoff ends with a touchback.")
                print(
                    f"{self.posteam.team_abbrev} starts drive at {self.yardline}"
                )
            self.start_drive()
        else:
            kick_distance = KickoffDistance_model.predict([self.qtr])[0]
            returned_from = kickoff_from - kick_distance
            if returned_from < -9:
                returned_from = -9
            return_yards = KickoffReturn_model.predict([self.qtr])[0]

            # Make sure return gets out of endzone
            if (returned_from + return_yards) <= 0:
                return_yards = np.abs(returned_from) + 5
            print(
                f"{self.defteam.team_abbrev} kicks to {self.posteam.team_abbrev} {returned_from} yard line"
            )
            penalty = KickoffReturnedPenalty_model.predict([self.qtr])[0]
            if (returned_from + return_yards) >= 100:
                return_yards = 100 - returned_from
                print(
                    f"{self.posteam.team_abbrev} returns ball {return_yards} yards for a touchdown!"
                )
                elapsed_time = KickoffReturnTD_elapsed_time_model.predict(
                    np.array(
                        [[return_yards, np.random.choice(range(0, 10000))]]
                    )
                )[0]
                self.score_change(self.posteam, 6)
                self.point_after_attempt(elapsed_time)
                return
            print(
                f"{self.posteam.team_abbrev} returns ball {return_yards} yards to {returned_from+return_yards} yardline"
            )
            if penalty == True:
                penalty_description = (
                    KickoffReturnedPenaltyDefteam_model.predict([self.qtr])[0]
                )
                penalty_type = " ".join(penalty_description.split(" ")[:-1])
                penalty_yards = float(penalty_description.split(" ")[-1])
                self.yardline = returned_from + return_yards + penalty_yards
                print(
                    f"{self.defteam.team_abbrev} comitted penalty {penalty_type}. {penalty_yards} yards added to the end of the play"
                )
                print(
                    f"{self.posteam.team_abbrev} starts drive at {self.yardline} yardline"
                )
                elapsed_time = KickoffReturn_elapsed_time_model.predict(
                    np.array(
                        [[return_yards, np.random.choice(range(0, 10000))]]
                    )
                )[0]
                self.elapsed_time(elapsed_time)
                self.start_drive()
                return
            fumble = KickoffFumble_model.predict([self.qtr])[0]
            if fumble == 1:
                fumble_lost = KickoffFumbleLost_model.predict([self.qtr])[0]
                if fumble_lost == 1:
                    fumble_td = KickoffFumbleLostReturnTD_model.predict(
                        [self.qtr]
                    )[0]
                    if fumble_td == True:
                        print(
                            f"{self.posteam.team_abbrev} fumbles on play. Fumble recovered by {self.defteam.team_abbrev} for a touchdown!"
                        )
                        self.score_change(self.defteam, 6)
                        self.change_possession()
                        self.point_after_attempt(3)

                    else:
                        fumble_yards = (
                            KickoffFumbleLostReturnYards_model.predict(
                                [self.qtr]
                            )[0]
                        )
                        if (
                            fumble_yards
                            + (100 - (returned_from + return_yards))
                        ) > 100:
                            fumble_yards = (returned_from + return_yards) - 1
                        self.yardline = 100 - (
                            returned_from + return_yards - fumble_yards
                        )
                        print(
                            f"{self.posteam.team_abbrev} fumbles on play. Fumble recovered by {self.defteam.team_abbrev} and returned {fumble_yards} yards to {self.yardline} yardline"
                        )
                        elapsed_time = (
                            KickoffReturnFumble_elapsed_time_model.predict(
                                np.array(
                                    [
                                        [
                                            return_yards,
                                            fumble_yards,
                                            np.random.choice(range(0, 10000)),
                                        ]
                                    ]
                                )
                            )[0]
                        )
                        self.change_possession()
                        self.elapsed_time(elapsed_time)
                        self.start_drive()
                else:
                    self.yardline = returned_from + return_yards
                    print(
                        f"{self.posteam.team_abbrev} fumbles on play but recovers at {self.yardline} yardline"
                    )
                    elapsed_time = KickoffReturn_elapsed_time_model.predict(
                        np.array(
                            [[return_yards, np.random.choice(range(0, 10000))]]
                        )
                    )[0]
                    self.elapsed_time(elapsed_time)
                    self.start_drive()
            else:
                self.yardline = returned_from = return_yards
                elapsed_time = KickoffReturn_elapsed_time_model.predict(
                    np.array(
                        [[return_yards, np.random.choice(range(0, 10000))]]
                    )
                )[0]
                self.elapsed_time(elapsed_time)
                self.start_drive()
                
    def field_goal(self):
        kick_distance = 117-self.yardline
        result = runModel([kick_distance],self.posteam.fg_model)
        elapsed_time = runModel([kick_distance],FG_ElapsedTime)
        if result=='made':
            print(f'{117-self.yardline} Field Goal Attempt is Good!')
            self.score_change(self.posteam,3)
            self.elapsed_time(elapsed_time)
            self.change_possession()
            self.kickoff()
        else:                          
            print(f'{117-self.yardline} Field Goal Attempt is No Good!')
            self.change_possession()
            self.elapsed_time(elapsed_time)
            self.yardline=100-self.yardline
            self.start_drive()
            
    def punt(self,safety=False):
        # If punt is  the result of a safety, set yardline to 25 and make sure
        # punt can't be blocked.
        if safety==True:
            self.yardline=25
            # Punt can't be blocked after safety
            result = 'blocked'
            while result=='blocked':
                result = PuntResult.predict([[self.qtr]])
        else:
            result = PuntResult.predict([[self.qtr]])
        if result == 'blocked':
            self.punt_blocked()
            return
        distance = runModel([self.yardline],PuntDistance)
        if result=='touchback':
            self.yardline=25
            elapsed_time = runModel([distance],PuntNonReturn_elapsed_time)
            self.elapsed_time(elapsed_time)
            self.change_possession()
            self.messageHandler(self,'punt_touchback',
                                self.defteam,
                                self.posteam,
                                self.game_seconds_remaining)
            self.start_drive()
        elif result in ['out_of_bounds','fair_catch']:
            self.yardline = 100 -(self.yardline+distance)
            if self.yardline<1:
                self.yardline=1
            elapsed_time = runModel([distance],PuntNonReturn_elapsed_time)
            self.elapsed_time(elapsed_time)
            self.change_possession()
            self.messageHandler(self,'punt_ob',
                                self.defteam,
                                self.posteam,
                                distance,
                                result,
                                self.yardline,
                                self.game_seconds_remaining)
            self.start_drive()
        elif result == 'muffed':
            self.muffed_punt(distance)
            return
        else:
            self.punt_returned(distance)
            
            
     
    def punt_blocked(self):
        result = PuntBlockResult.predict([[self.yardline]])
        if result == 'touchdown':
            elapsed_time = PuntBlockTD_elapsed_time.predict([[self.yardline]])
            self.score_change(self.defteam,6)
            self.change_posession()
            self.messageHandler(self,'punt_block_return_td',
                                self.defteam,
                                self.posteam,
                                self.game_seconds_remaining)
            self.point_after_attempt(elapsed_time)
        elif result == 'safety':
            elapsed_time = PuntBlockSafety_elapsed_time.predict([[self.yardline]])
            self.score_change(self.defteam,2)
            self.elapsed_time(elapsed_time)
            self.change_possession()
            self.messageHandler(self,'punt_block_saftey',
                                self.defteam,
                                self.posteam,
                                self.game_seconds_remaining)
            self.punt(safety=True)
            self.start_drive()
        elif result == 'punt_block_out_of_bounds':
            elapsed_time = PuntBlockOB_elapsed_time.predict([[self.yardline]])
            ob_at = runModel([self.yardline],PuntBlockRecoveredAt)
            self.elapsed_time(elapsed_time)
            self.yardline = 100-ob_at
            self.change_possession()
            self.messageHandler(self,'punt_block_ob',
                                self.defteam,
                                self.posteam,
                                ob_at,
                                self.game_seconds_remaining)
            self.start_drive()
        else:
            # Who Recovered Block?
            block_recovered_by = PuntBlockRecoveredBy.predict([[self.qtr]])
            # Where was Block Recovered?
            block_recovered_at = runModel([self.yardline],PuntBlockRecoveredAt)
            # How Many return yards?
            return_yards = runModel([1,block_recovered_by,block_recovered_at],PuntBlockReturnYards)
            # How much time passed?
            elapsed_time = runModel([self.yardline,
                                      return_yards],PuntBlockReturn_elapsed_time)
            # If Defense Recovered
            if block_recovered_by == True:
                # Was block returned for a touchdown?
                if (block_recovered_at - return_yards)<=0:
                    elapsed_time = PuntBlockTD_elapsed_time.predict([[self.yardline]])
                    self.score_change(self.defteam,6)
                    self.change_posession()
                    self.messageHandler(self,'punt_block_return_td',
                                        self.defteam,
                                        self.posteam,
                                        self.game_seconds_remaining)
                    self.point_after_attempt(elapsed_time)
                else:
                    self.elapsed_time(elapsed_time)
                    self.yardline = 100-(block_recovered_at - return_yards)
                    self.change_possession()
                    self.messageHandler(self,'punt_block_return',
                                        self.defteam,
                                        self.posteam,
                                        recovered_by,
                                        recovered_at,
                                        return_yards,
                                        self.yardline,
                                        self.game_seconds_remaining)
                    self.start_drive()
            # If Possession Team recovered
            else:
                # See if block was recovered and ran for a first down
                self.ydstogo = self.ydstogo + (self.yardline-block_recovered_at)
                if return_yards>=self.ydstogo:    
                    self.yardline = block_recovered_at + return_yards
                    self.elapsed_time(elapsed_time)
                    self.messageHandler(self,'punt_block_return',
                                        self.posteam,
                                        self.defteam,
                                        recovered_by,
                                        recovered_at,
                                        return_yards,
                                        self.yardline,
                                        self.game_seconds_remaining)
                    self.start_drive()
                else:
                    self.yardline = 100 - (block_recovered_at + return_yards)
                    self.elapsed_time(elapsed_time)
                    self.change_possession()
                    self.messageHandler(self,'punt_block_return',
                                        self.defteam,
                                        self.posteam,
                                        recovered_by,
                                        recovered_at,
                                        return_yards,
                                        self.yardline,
                                        self.game_seconds_remaining)
                    self.start_drive()
                    
    def muffed_punt(self,distance):
        # QC Distance
        if (self.yardline+distance)>=100:
            distance = 99-self.yardline
        # Punt was muffed, who recovered?
        fumble_lost = runModel([self.qtr],PuntReturnFumbleLost)
        
        # If fumble recovered, proceed to punt returned
        if fumble_lost==0:
            self.punt_returned(distance)
        else:
            return_yards = runModel([self.qtr],PuntReturnDefFumbleYards)
            if (self.yardline+distance+return_yards)>=100:
                # Touchdown
                return_yards = 100-(self.yardline+distance)
                self.score_change(self.defteam,6)
                self.change_possession()
                elapsed_time=runModel([distance,return_yards],PuntReturnDefFumbleTD_elapsed_time)
                self.messageHandler(self,'muffed_punt_td',
                                    self.defteam,
                                    self.posteam,
                                    self.game_seconds_remaining)
                self.point_after_attempt(elapsed_time)
            else:
                self.yardline = 100 - (self.yardline+distance+return_yards)
                elapsed_time = runModel([distance,return_yards],PuntReturnFumbleLost_elapsed_time)
                self.change_possession(elapsed_time)
                self.messageHandler(self,'muffed_punt_return',
                                    self.defteam,
                                    self.posteam,
                                    return_yards,
                                    self.yardline,
                                    self.game_seconds_remaining)
                self.start_drive()
    
    def punt_returned(self,distance):
        if (self.yardline+distance)>=100:
            distance = 99-self.yardline
        self.yardline = 100 - (self.yardline+distance)
        return_yards = runModel([self.qtr],PuntReturnYards)
        elapsed_time = runModel([distance,return_yards],PuntReturn_elapsed_time)
        # See if punt returned for touchdown
        if (self.yardline+distance) >=100:
            distance = 100-self.yardline
            self.score_change(self.defteam,6)
            self.change_possession()
            self.messageHandler(self,'punt_return_td',
                                self.defteam,
                                self.posteam,
                                return_yards,
                                self.game_seconds_remaining)
            self.point_after_attempt(elapsed_time)
        else:
            self.yardline+=distance
            self.change_possession()
            self.messageHandler(self,'punt_return',
                                self.defteam,
                                self.posteam,
                                return_yards,
                                self.yardline,
                                self.game_seconds_remaining)
            self.start_drive()
            

    def point_after_attempt(self, elapsed_time):
        if self.end_game == True:
            return
        print("Extra Point Attempt")
        two_points = self.two_point_attempt_decision()
        two_points = False
        if two_points == True:
            print(f"{self.posteam.team_abbrev} Attempts 2 Point Conversion")
            self.run_two_point_conversion()
        else:
            self.extra_point()
        self.elapsed_time(elapsed_time)
        self.change_possession()
        if self.end_game != True:
            self.kickoff()
        else:
            return

    def score_change(self, scoring_team, points):
        if scoring_team == self.home_team:
            self.home_team_score += points
        else:
            self.away_team_score += points
        if (self.overTime == True) & (points in [2, 6]):
            self.end_game = True
            self.elapsed_time(600)
        if (self.overTime == True) & (self.OT_first_possession == False):
            if self.home_team_score != self.away_team_score:
                self.end_game = True
                self.elapsed_time(600)

    def change_possession(self):
        if self.home_team == self.posteam:
            self.posteam = self.away_team
            self.posteam_score = self.away_team_score
            self.away_team_timeouts_remaining = self.defteam_timeouts_remaining 
            self.defteam = self.home_team
            self.defteam_score = self.home_team_score
            self.home_team_timeouts_remaining =self.posteam_timeouts_remaining 
        else:
            self.posteam = self.home_team
            self.posteam_score = self.home_team_score
            self.home_team_timeouts_remaining =self.defteam_timeouts_remaining
            self.defteam = self.away_team
            self.defteam_score = self.away_team_score
            self.away_team_timeouts_remaining =self.posteam_timeouts_remaining
        if self.overTime == True:
            self.OT_first_possession = False

    def elapsed_time(self, time):
        self.game_seconds_remaining -= time
        self.half_seconds_remaining -= time

        if self.half_seconds_remaining<=120:
            if self.half==1:
                self.posteam_timeout=FirstHalfPosteam2MinTimeout
                self.defteam_timeout=FirstHalfDefteam2MinTimeout
            else:
                self.posteam_timeout=SecondHalfPosteam2MinTimeout
                self.defteam_timeout=SecondHalfDefteam2MinTimeout
        if self.half_seconds_remaining <= 0:
            print(f"End of Half {self.half}")
            self.half += 1
            self.half_seconds_remaining = 1800
            if self.half==2:
                self.qtr += 1
                self.qtr_seconds_remaining = 900
                self.home_timeouts_remaining = 3
                self.away_timeouts_remaining = 3
                self.posteam_timeouts_remaining=3
                self.defteam_timeouts_remaining=3
                self.posteam_timeout = SecondHalfPosteamTimeout
                self.defteam_timeout = SecondHalfDefteamTimeout
                self.kickoff()
        self.qtr_seconds_remaining -= time
        if self.qtr_seconds_remaining <= 0:
            print(f"End of Quarter {self.qtr}")
            self.qtr += 1
            self.qtr_seconds_remaining = 900
            if self.qtr==4:
                self.posteam.RunPass_model=self.posteam.RunPass4thQtr_model
                self.posteam.FourthDownDecisionModel = self.posteam.FourthDown4thQtrDecisionModel
                self.defteam.RunPass_model = self.defteam.RunPass4thQtr_model
                self.defteam.FourthDownDecisionModel = self.defteam.FourthDown4thQtrDecisionModel
                RushModels=RushModels4thQtr
        if self.game_seconds_remaining <= 0:
            if self.home_team_score == self.away_team_score:
                if self.qtr == 5:
                    print("Going into overtime!")
                    self.game_seconds_remaining = 600
                    self.qtr_seconds_remaining = 600
                    self.half_seconds_remaining = 600
                    self.home_timeouts_remaining = 2
                    self.away_timeouts_remaining = 2
                    self.posteam_timeouts_remaining=2
                    self.defteam_timeouts_remaining=2
                    self.qtr = 5
                    self.overTime = True
                    self.coinToss()
                    self.kickoff()
            else:
                self.stats['home_team_score']=self.home_team_score
                self.stats['away_team_score']=self.away_team_score
                self.stats['iteration']=self.iteration
                self.export_stats(self.stats,self.iteration)
                self.end_game = True
                if self.home_team_score > self.away_team_score:
                    winner = self.home_team.team_abbrev
                    winning_score = self.home_team_score
                    losing_score = self.away_team_score
                else:
                    winner = self.away_team.team_abbrev
                    winning_score = self.away_team_score
                    losing_score = self.home_team_score
                if self.overTime == True:
                    print(
                        f"{winner} wins {winning_score} - {losing_score} in over time"
                    )
                else:
                    print(
                        f"{winner} wins {winning_score} - {losing_score}!"
                    )
                sys.exit()

    def two_point_attempt_decision(self):
        if self.qtr < 4:
            prob_no = TwoPointDecision_model.predict_proba(
                np.array(
                    [
                        [
                            self.posteam_score - self.defteam_score,
                            self.game_seconds_remaining,
                        ]
                    ]
                )
            )[0][0]
            random_choice = np.random.choice(np.arange(0, 1, 0.01))
            if random_choice < prob_no:
                return False
            else:
                return True
        else:
            return TwoPointDecision_model.predict(
                np.array(
                    [
                        [
                            self.posteam_score - self.defteam_score,
                            self.game_seconds_remaining,
                        ]
                    ]
                )
            )[0]

    def extra_point(self):
        kick_from = XPFrom_model.predict([self.qtr])[0]
        kick_result = runModel([kick_from],XP_model)
        if kick_result=='blocked':
            safety = XPBlockedReturn_model.predict([self.qtr])[0]
            if safety == True:
                self.score_change(self.defteam, 2)
                self.change_possession()
                self.punt()
        elif kick_result=='failed':
            pass
        else:
            self.score_change(self.posteam, 1)
        print(
            f"{self.posteam.team_abbrev} kicks extra point from {kick_from} yardline. Extra point attempt {kick_result}"
        )
        
    def start_drive(self):
        if self.end_game == True:
            return
        self.down = 1
        self.ydstogo = 10
        self.play_clock = 40
        self.clock_running = False
        self.score_differential = self.posteam_score - self.defteam_score
        self.run_play(self.down,self.ydstogo)


    def run_play(self,down,to_go):
        if self.end_game == True:
            return
        self.pre_snap_penalty()
        self.down=down
        self.ydstogo=to_go
        self.play_frame = stat_frame(self)
        self.play_frame['posteam']=[str(self.posteam)]
        self.play_frame['defteam']=[str(self.defteam)]
        if self.down==4:
            self.play_type = runModel(                    [
                                    self.ydstogo,
                                    self.yardline,
                                    self.half_seconds_remaining,
                                    self.game_seconds_remaining,
                                    self.score_differential,
                                ], self.posteam.FourthDownRunPass_model)
            if self.play_type == "pass":
                play_type = "Pass"
                self.formation = runModel(                            [
                                        self.ydstogo,
                                        self.yardline,
                                        self.half_seconds_remaining,
                                        self.game_seconds_remaining,
                                        self.score_differential,
                                            ],self.posteam.FourthDownPassFormation_model)
                self.personnel_model = self.posteam.FourthDownPassFormation_personnel_model


                self.n_pass_rushers=runModel(                        [
                                            self.yardline,
                                            self.down,
                                            self.ydstogo,
                                            self.half_seconds_remaining,
                                            self.score_differential,
                                        ],self.defteam.n_pass_rushers_model)
            else:
                play_type = "Run"
                self.formation = runModel(                            [
                                            self.yardline,
                                            self.down,
                                            self.ydstogo,
                                            self.half_seconds_remaining,
                                            self.score_differential,
                                            ],self.posteam.FourthDownRunFormation_model)
                self.personnel_model = self.posteam.FourthDownRunFormation_personnel_model
                self.n_pass_rushers = None
            if len(self.posteam.FourthDownRunFormation_personnel_model.classes_)==1:
                self.personnel=self.posteam.FourthDownRunFormation_personnel_model.classes_[0]
            else:
                self.personnel = runModel([self.down, self.ydstogo],
                                          self.personnel_model)
            
        else:
            self.play_type = runModel(                    [
                                    self.qtr,
                                    self.down,
                                    self.ydstogo,
                                    self.yardline,
                                    self.half_seconds_remaining,
                                    self.game_seconds_remaining,
                                    self.score_differential,
                                ],self.posteam.RunPass_model)
            if self.play_type == "pass":
                play_type = "Pass"
                self.formation = runModel([self.down,
                                          self.ydstogo,
                                          self.half_seconds_remaining,
                                          self.score_differential,
                                            ],self.posteam.PassFormation_model)
                self.n_pass_rushers = runModel([self.yardline,
                                                self.down,
                                                self.ydstogo,
                                                self.half_seconds_remaining,
                                                self.score_differential],
                                                self.defteam.n_pass_rushers_model)
            else:
                play_type = "Run"
                self.formation = runModel([self.down,
                                           self.ydstogo,
                                           self.half_seconds_remaining,
                                           self.score_differential,
                                            ],self.posteam.RunFormation_model)
                self.n_pass_rushers = None
            if len(self.posteam.Personnel_models[play_type][self.formation].classes_)==1:
                self.personnel=self.posteam.Personnel_models[play_type][self.formation].classes_[0]
            else:
                self.personnel = runModel([self.down, self.ydstogo],
                                          self.posteam.Personnel_models[play_type][self.formation])
        # Get Defensive Formation (Personnel, defenders in box, number of pass rushers)
        self.defense_personnel = runModel([
                                self.yardline,
                                self.down,
                                self.ydstogo,
                                self.half_seconds_remaining,
                                self.score_differential,
                            ],self.defteam.defense_personnel_model)
        self.defenders_in_box = runModel(
                            [
                            self.yardline,
                            self.down,
                            self.ydstogo,
                            self.half_seconds_remaining,
                            self.score_differential,
                            ],self.defteam.defenders_in_box_model)

        self.lineup_players(self.personnel, self.defense_personnel)
        if self.play_type=='run':
            self.play_frame['rush_attempt']=[1]
            self.play_frame['pass_attempt']=[np.nan]
            self.play_frame['complete_pass']=[np.nan]
            self.play_frame['passer_player_id']=[np.nan]
            self.play_frame['receiver_player_id']=[np.nan]
            self.play_frame['passing_yards']=[np.nan]
            self.play_frame['pass_touchdown']=[np.nan]
            self.play_frame['interception']=[np.nan]
            self.play_frame['sack']=[np.nan]
            #self.pre_snap_penalty()
            self.executeRunPlay()
        elif self.play_type=='pass':
            self.play_frame['pass_attempt']=[1]
            self.play_frame['rush_attempt']=[np.nan]
            self.play_frame['rusher_player_id']=[np.nan]
            self.play_frame['rushing_yards']=[np.nan]
            self.play_frame['rush_touchdown']=[np.nan]
            self.executePassPlay()
        elif self.play_type=='qb_kneel':
            self.play_frame['pass_attempt']=[np.nan]
            self.play_frame['rush_attempt']=[1]
            self.play_frame['rusher_player_id']=[np.nan]
            self.play_frame['rushing_yards']=[np.nan]
            self.play_frame['rush_touchdown']=[np.nan]
            self.QBKneel()
        elif self.play_type=='qb_spike':
            self.play_frame['pass_attempt']=[1]
            self.play_frame['rush_attempt']=[np.nan]
            self.play_frame['rusher_player_id']=[np.nan]
            self.play_frame['rushing_yards']=[np.nan]
            self.play_frame['rush_touchdown']=[np.nan]
            self.QBSpike()

    def QBKneel(self):
        self.ydstogo+=2
        self.yardline-=2
        self.down+=1
        print(f'{self.posteam} kneels ball at {self.yardline}')
        elapsed_time = FourthQtrQBKneelElapsedTime.predict([[self.qtr]])
        self.elapsed_time(elapsed_time)
        self.clock_running=True
        if self.down<4:
            self.run_play(self.down,self.ydstogo)
        elif self.down==4:
            self.FourthDownDecisionTree()
        else:
            self.clock_running=False
            self.change_posession()
            self.start_drive()
    
    def QBSpike(self):
        if self.down==4:
            self.run_play(self.down,self.ydstogo)
        self.ydstogo+=1
        self.yardline-=1
        self.down+=1
        print(f'{self.posteam} spikes ball at {self.yardline}')
        self.elapsed_time(2)
        self.clock_running=False
        if self.down<4:
            self.run_play(self.down,self.ydstogo)
        elif self.down==4:
            self.FourthDownDecisionTree()

    def lineup_players(self, personnel, d_personnel):
        for position in personnel.split(","):
            if "RB" in position:
                rbs = int(position.split("RB")[0])
            elif "WR" in position:
                wrs = int(position.split("WR")[0])
            elif "TE" in position:
                tes = int(position.split("TE")[0])
        ols = 11 - (rbs + wrs + tes + 1)
        self.OL = selectOffensivePlayers(
            self, 'OL', ols
        )
        self.QB = selectOffensivePlayers(
            self, 'QB', 1
        )
        self.play_frame['passer_player_id'] = [self.QB.full_name.values[0]]
        self.RB = selectOffensivePlayers(
            self,'RB', rbs
        )
        self.WR = selectOffensivePlayers(
            self,'WR', wrs
        )
        self.TE = selectOffensivePlayers(
            self,'TE', tes
        )

        for position in d_personnel.split(","):
            if "DB" in position:
                dbs = int(position.split("DB")[0])
            elif "LB" in position:
                lbs = int(position.split("LB")[0])
            elif "DL" in position:
                dls = int(position.split("DL")[0])
        self.DL = selectDefensivePlayers(
            self,self.defteam.depth_chart["DL"], dls
        )
        self.LB = selectDefensivePlayers(
            self,self.defteam.depth_chart["LB"], lbs
        )
        self.DB = selectDefensivePlayers(
            self,self.defteam.depth_chart["DB"], dbs
        )
        return


    
    def executePassPlay(self):
        sacked = self.runSackModel()   
        if sacked==True:
            self.clock_running==True
            self.play_frame['sack']=[1]
            self.play_frame['interception']=[np.nan]
            self.play_frame['pass_attempt']=[np.nan]
            self.play_frame['complete_pass']=[np.nan]

            self.play_frame['passing_yards']=[np.nan]
            self.play_frame['passer_player_id']=[np.nan]
            self.play_frame['receiver_player_id']=[np.nan]
            self.play_frame['pass_touchdown']=[np.nan]
            self.stats =pd.concat([self.stats,self.play_frame])
            if self.down<4:
                self.run_play(self.down,self.ydstogo)
            else:
                self.FourthDownDecisionTree()
        else:
            self.air_yards, self.pass_length = determinePassLength(self)
            self.receiver = determineReceiver(self,self.pass_length)
            self.receiver_player_id = self.receiver.full_name.values[0]
            self.play_frame['receiver_player_id']=[self.receiver_player_id]
            self.passModel = getPassModel(self)
            pass_features = retrievePassFeatures(self,self.pass_length,self.air_yards)
            result = runModel(pass_features.values[0],self.passModel)
            # Incomplete Pass Result
            if result==0:
                self.clock_running=False
                self.play_frame['passing_yards']=[0]
                self.play_frame['pass_touchdown']=[0]
                self.play_frame['complete_pass']=[0]
                self.play_frame['interception']=[0]
                self.play_frame['sack']=[0]
                self.stats =pd.concat([self.stats,self.play_frame])
                elapsed_time = runModel([self.air_yards],
                                        IncompletePassElapsedTime)
                self.pass_et.append(elapsed_time)
                self.down+=1
                if self.down<4:
                    print(f'{self.QB.full_name.values[0]} pass intended for {self.receiver.full_name.values[0]}. Pass Incomplete. {self.down} and {self.ydstogo}')
                    self.elapsed_time(elapsed_time)
                    self.run_play(self.down,self.ydstogo)
                elif self.down==4:
                    print(f'{self.QB.full_name.values[0]} pass intended for {self.receiver.full_name.values[0]}. Pass Incomplete. {self.down} and {self.ydstogo}')
                    self.elapsed_time(elapsed_time)
                    self.FourthDownDecisionTree()
                else:
                    print(f'{self.QB.full_name.values[0]} pass intended for {self.receiver.full_name.values[0]}. Pass Incomplete. Turnover on Downs!')
                    self.elapsed_time(elapsed_time)
                    self.change_possession()
                    self.yardline = 100-self.yardline
                    self.start_drive()
            #Complete Pass Result
            elif result==1:
                # getYAC
                # Checickoff if pass was in end zone
                self.play_frame['complete_pass']=[1]
                self.play_frame['interception']=[0]
                self.play_frame['sack']=[0]
                self.clock_running=True
                if self.yardline+self.air_yards >=100:
                    self.clock_running=False
                    pass_yards = 100-self.yardline
                    self.play_frame['passing_yards']=[pass_yards]
                    self.play_frame['pass_touchdown']=[1]
                    self.stats =pd.concat([self.stats,self.play_frame])
                    print(f'{self.QB.full_name.values[0]} pass {self.air_yards} yards to {self.receiver.full_name.values[0]}. Touchdown!')
                    elapsed_time = runModel([self.air_yards,0,
                                             0,self.clock_running],
                                            CompletePassElapsedTime)
                    self.pass_et.append(elapsed_time)
                    self.score_change(self.posteam, 6)
                    self.point_after_attempt(elapsed_time)
                # If Pass wasn't in endzone, determine YAC
                # See if player went out of bounds after play
                else:
                    yac_features =retrieveYacFeatures(self,self.air_yards)
                    yac = runModel(yac_features.values[0],YACModel)
                    ob = runModel([
                        self.down,
                        self.ydstogo,
                        self.yardline,
                        self.half_seconds_remaining,
                        self.air_yards,
                        yac
                        ],RecOB)
                    if (ob==True)&(((self.qtr==2)&(self.half_seconds_remaining<=120))|((self.qtr==4)&(self.half_seconds_remaining<=300))):
                        self.clock_running=False
                    self.timeout()

                    # Check if player caught ball and ran to end zone
                    if self.yardline+self.air_yards+yac >=100:
                        print(f'{self.QB.full_name.values[0]} pass {self.air_yards+yac} yards to {self.receiver.full_name.values[0]}. Touchdown!')

                        yac = 100-(self.yardline+self.air_yards)
                        pass_yards=self.air_yards+yac
                        elapsed_time = runModel([self.air_yards,yac,
                                                 ob,self.clock_running],
                                                CompletePassElapsedTime)
                        self.play_frame['passing_yards']=[pass_yards]
                        self.play_frame['pass_touchdown']=[1]
                        self.stats =pd.concat([self.stats,self.play_frame])

                        elapsed_time = runModel([self.air_yards,yac,
                                                 ob,self.clock_running],
                                                CompletePassElapsedTime)
                        self.pass_et.append(elapsed_time)

                        self.score_change(self.posteam,6)
                        self.point_after_attempt(elapsed_time)
                        return
                    # Player caught ball and ran for first down
                    self.timeout()
                    elapsed_time = runModel([self.air_yards,yac,
                                             ob,self.clock_running],
                                            CompletePassElapsedTime)
                    if (self.air_yards+yac >= self.ydstogo):
                        print(f'{self.QB.full_name.values[0]} pass {self.air_yards+yac} yards to {self.receiver.full_name.values[0]}. First Down!')
                        pass_yards=self.air_yards+yac
                        self.play_frame['passing_yards']=[pass_yards]
                        self.play_frame['pass_touchdown']=[0]
                        self.stats =pd.concat([self.stats,self.play_frame])
                        self.elapsed_time(elapsed_time)
                        self.pass_et.append(elapsed_time)
                        self.yardline+=(self.air_yards+yac)
                        self.stats =pd.concat([self.stats,self.play_frame])
                        self.run_play(down=1,to_go=10)
                    else:
                        pass_yards=self.air_yards+yac
                        self.play_frame['passing_yards']=[pass_yards]
                        self.play_frame['pass_touchdown']=[0]
                        self.elapsed_time(elapsed_time)
                        self.pass_et.append(elapsed_time)
                        self.yardline+=(self.air_yards+yac)
                        print(f'{self.QB.full_name.values[0]} pass {self.air_yards+yac} yards to {self.receiver.full_name.values[0]}. {self.down+1} and {self.ydstogo-(self.air_yards+yac)}!')
                        self.stats =pd.concat([self.stats,self.play_frame])
                        if self.down<4:
                            self.run_play(self.down+1,self.ydstogo-(self.air_yards+yac))
                        else:
                            self.FourthDownDecisionTree()
            # Pass Intercepted
            else:
                self.clock_running=False
                print(f'{self.QB.full_name.values[0]} pass intended for {self.receiver.full_name.values[0]}. Pass Intercepted!')
                self.play_frame['interception']=[1]
                self.play_frame['sack']=[0]
                self.play_frame['passing_yards']=[0]
                self.play_frame['pass_touchdown']=[0]
                self.play_frame['complete_pass']=[0]
                return_yards = runModel([
                    self.down,
                    self.ydstogo,
                    self.yardline,
                    self.score_differential,
                    self.half_seconds_remaining,
                ],InterceptionReturnYardsModel)
                elapsed_time = runModel(                        [
                                            self.down,
                                            self.ydstogo,
                                            self.yardline,
                                            self.score_differential,
                                            self.half_seconds_remaining,
                                        ],PassIntElapsedTime
                    )
                self.pass_et.append(elapsed_time)

                # Check for return touchdown
                if (self.yardline+self.air_yards-return_yards)<=0:
                    return_yards = (self.yardline+self.air_yards)
                    self.score_change(self.defteam,6)
                    self.change_possession()
                    self.point_after_attempt(elapsed_time)
                else:
                    self.yardline=100-((self.yardline+self.air_yards)-return_yards)
                    self.elapsed_time(elapsed_time)
                    self.change_possession()
                    self.start_drive()
                return
            
    def runSackModel(self):
        sack_features = retrieveSackFeatures(self)
        self.sacked = runModel(sack_features.values[0],sackModel)
        if self.sacked==True:
            self.clock_running=True
            sack_yards = runModel(sack_features.values[0],sackYardsModel)
            # sack_yards=np.random.choice(range(10))
            # Check for strip Sack
            stripSacked = runModel(sack_features.values[0],stripSackModel)
            if stripSacked==True:
                self.clock_running=False
                # Check if strip sack occurred in end zone
                return_yards = runModel([self.down],stripSackReturnYardsModel)
                if self.yardline-sack_yards <=0:
                    return_yards=0
                    return_touchdown=1
                    sack_yards = self.yardline
                    print(f'{self.posteam} QB fumbled in endzone. Recovered by {self.defteam} for a touchdown!')
                    self.score_change(self.defteam, 6)
                    elapsed_time = runModel([return_yards,
                                             return_touchdown,
                                             sack_yards*-1
                                             ],stripSackElapsedTimeModel)
                    self.pass_et.append(elapsed_time)
                    self.messageHandler(self,
                                        'strip_sack_td',
                                        self.posteam,
                                        self.defteam,
                                        self.game_seconds_remaining-elapsed_time
                                        )
                    self.change_possession()
                    self.point_after_attempt(elapsed_time)
                # Check if strip sack was returned for touchdown
                elif (self.yardline - sack_yards - return_yards )<=0:
                    return_yards = self.yardline-self.sack_yards
                    return_touchdown = 1
                    print(f'{self.posteam} QB fumbled. Recovered by {self.defteam} and returned {return_yards} yards for a touchdown!')
                    elapsed_time = runModel([return_yards,
                                             return_touchdown,
                                             sack_yards*-1
                                             ],stripSackElapsedTimeModel)
                    self.pass_et.append(elapsed_time)
                    self.score_change(self.defteam, 6)
                    self.change_possession()
                    self.point_after_attempt(elapsed_time)
                    
                else:
                    return_touchdown=0
                    print(f'{self.posteam} QB Fumbled. Recovered by {self.defteam} and returned {return_yards} yards')
                    elapsed_time = runModel([return_yards,
                                             return_touchdown,
                                             sack_yards*-1
                                             ],stripSackElapsedTimeModel)
                    self.yardline=100-(self.yardline-sack_yards)
                    self.elapsed_time(elapsed_time)
                    self.pass_et.append(elapsed_time)
                    self.change_possession()
                    self.start_drive()
            else:

                # Check if sack was a safety
                if self.yardline - sack_yards <=0:
                    self.clock_running=False
                    elapsed_time = runModel([self.down,
                                             self.ydstogo,
                                             self.yardline,
                                             self.score_differential,
                                             self.half_seconds_remaining,
                                             sack_yards*-1
                                             ],sackElapsedTimeModel)
                    sack_yards=self.yardline
                    print(f'{self.posteam} QB sacked in endzone for a safety!')
                    self.score_change(self.defteam,2)
                    self.elapsed_time(elapsed_time)
                    # Insert Punt Code Later
                    self.change_possession()
                    self.yardline=30
                    self.start_drive()
                else:

                    self.yardline-=sack_yards
                    self.ydstogo+=sack_yards
                    self.down+=1
                    if self.down>4:
                        self.clock_running=False
                        elapsed_time = runModel([self.down,
                                                 self.ydstogo,
                                                 self.yardline,
                                                 self.score_differential,
                                                 self.half_seconds_remaining,
                                                 sack_yards*-1
                                                 ],sackElapsedTimeModel)
                        print(f'{self.posteam} QB sacked. Turnover on downs!')
                        self.elapsed_time(elapsed_time)
                        self.change_possession()
                        self.yardline = 100-self.yardline
                        self.start_drive()
                    else:
                        self.timeout()
                        elapsed_time = runModel([self.down,
                                                 self.ydstogo,
                                                 self.yardline,
                                                 self.score_differential,
                                                 self.half_seconds_remaining,
                                                 sack_yards*-1
                                                 ],sackElapsedTimeModel)
                        print(f'{self.posteam} QB sacked for a loss of {sack_yards} yards')
                        return self.sacked
        else:
            self.sacked=False
            return self.sacked
    


    def executeRunPlay(self):
        runner,runpos = determineRunner(self)
        self.rusher_player_id = runner.full_name.values[0]
        rush_model,rush_et_model,rush_ob_model=RushModels[runpos]
        rush_features = retrieveRushFeatures(self,runner)
        rush_yards = runModel(rush_features.values[0],rush_model)
        self.play_frame['rusher_player_id']=[self.rusher_player_id]
        # Check if player scored a touchdown
        if self.yardline+rush_yards>=100:
            self.clock_running=False
            rush_yards=100-self.yardline
            self.play_frame['rush_touchdown']=[1]
            self.play_frame['rushing_yards']=[rush_yards]
            self.stats =pd.concat([self.stats,self.play_frame])
            elapsed_time = runModel([rush_yards,
              self.down,
              self.ydstogo,
              self.yardline,
              self.half_seconds_remaining,
              self.clock_running,
              0
              ],rush_et_model)
            print(f'{runner.full_name.values[0]} rushes {rush_yards} yards for a touchdown!')
            self.score_change(self.posteam, 6)
            self.point_after_attempt(elapsed_time)
            return
        self.play_frame['rush_tochdown']=[np.nan]
        self.play_frame['rushing_yards']=[rush_yards]

        self.stats =pd.concat([self.stats,self.play_frame])

        # if player didn't score, check if player lost a fumble
        fumbled = runModel([rush_yards,
          self.down,
          self.ydstogo,
          self.yardline,
          self.half_seconds_remaining],RushFumbleModel)
        # If player lost a fumble, determine def return yards
        fumbled=0
        if fumbled==1:
            self.clock_running=False
            fumbled_at = self.yardline+rush_yards
            return_yards = runModel([rush_yards,
              self.down,
              self.ydstogo,
              self.yardline,
              self.half_seconds_remaining],RushFumbleReturnYardsModel)
            # Check if fumble was returned for touchdown
            if fumbled_at - return_yards <=0:
                return_yards = fumbled_at
                # Get elapsed time for fumble play
                elapsed_time = runModel([rush_yards,
                  self.down,
                  self.ydstogo,
                  self.yardline,
                  self.half_seconds_remaining,
                  return_yards],RushFumbleElapsedTime)
                print(f'{self.defteam} recovers fumble and returns for {return_yards} yards for a touchdown!')

                self.score_change(self.defteam, 6)
                self.change_possession()
                self.point_after_attempt(elapsed_time)
                return
            else:
                # Get elapsed time for fumble play
                elapsed_time = runModel([rush_yards,
                  self.down,
                  self.ydstogo,
                  self.yardline,
                  self.half_seconds_remaining,
                  return_yards],RushFumbleElapsedTime)
                self.yardline = 100 - (fumbled_at - return_yards)
                print(f'{self.defteam} recovers fumble and returns for {return_yards} yards to {self.yardline}!')
                self.elapsed_time(elapsed_time)
                self.change_possession()
                self.start_drive()
                return
        # If Player Didn't fumble or score a touchdown, continue to next down
        else:
            self.clock_running=True
            # Did player run out of bounds?
            ob=runModel([rush_yards,
                         self.down,
                         self.ydstogo,
                         self.yardline,
                         self.half_seconds_remaining],RushOB)
            if (ob==True)&((self.qtr==2)&(self.half_seconds_remaining<=120))|((self.qtr==4)&(self.half_seconds_remaining<=300)):
                    self.clock_running=False
            self.timeout()
            if self.ydstogo-rush_yards<=0:
                elapsed_time = runModel([rush_yards,
                  self.down,
                  self.ydstogo,
                  self.yardline,
                  self.half_seconds_remaining,
                  self.clock_running,
                  ob],rush_et_model)
                self.yardline+=rush_yards
                print(f'{runner.full_name.values[0]} rushes {rush_yards} yards to {self.yardline} yardline for a first down!')
                self.elapsed_time(elapsed_time)
                self.run_play(down=1,to_go=10)
            else:
                elapsed_time = runModel([rush_yards,
                  self.down,
                  self.ydstogo,
                  self.yardline,
                  self.half_seconds_remaining,
                  self.clock_running,
                  ob],rush_et_model)
                self.elapsed_time(elapsed_time)
                self.yardline+=rush_yards
                if self.down==4:
                    self.FourthDownDecisionTree()                        
                elif self.down==5:
                    print('Turnover on downs!')
                    self.clock_running=False
                    self.change_possession()
                    self.yardline = 100-self.yardline
                    self.start_drive()
                else:
                    print(f'{runner.full_name.values[0]} rushes {rush_yards} yards to {self.yardline} yardline. {self.down} and {self.ydstogo}')
                    self.run_play(self.down+1,self.ydstogo-rush_yards)

    def FourthDownDecisionTree(self):
        go_for_it = runModel(                [
                        self.ydstogo,
                        self.yardline,
                        self.half_seconds_remaining,
                        self.game_seconds_remaining,
                        self.score_differential
                        ],self.posteam.FourthDownDecisionModel)
        if go_for_it ==True:
            self.run_play(self.down,self.ydstogo)
        else:
            fg_or_punt = runModel([
                                    self.yardline,
                                    self.down,
                                    self.ydstogo,
                                    self.half_seconds_remaining,
                                    self.score_differential],self.posteam.punt_fg_model)
            if fg_or_punt == 'punt':
                print(f'{self.posteam} Punts')
                self.punt()
            else:
                self.field_goal()
    
    def timeout(self):
        if self.clock_running==False:
            return
        # Check if posteam wants to take a timeout
        if self.posteam_timeouts_remaining>0:
            timeout = runModel([self.half_seconds_remaining,
                                self.down+1,
                                self.ydstogo,
                                self.yardline,
                                ],self.posteam_timeout)
            if timeout==True:
                self.clock_running==False
                self.posteam_timeouts_remaining-=1
                print(f'{self.posteam} takes a timeout. {self.posteam_timeouts_remaining} timeouts remaining')

                return
        # Check if defteam wants to take a timeout
        if self.defteam_timeouts_remaining>0:
            timeout = runModel([self.half_seconds_remaining,
                                self.down+1,
                                self.ydstogo,
                                self.yardline,
                                ],self.defteam_timeout)
            if timeout==True:
                self.clock_running==False
                self.defteam_timeouts_remaining-=1
                print(f'{self.defteam} takes a timeout. {self.defteam_timeouts_remaining} timeouts remaining')

                return    
    
    def pre_snap_penalty(self):
        
                        
#%%
# schedule = nfl.import_schedules(range(2017, 2023))
# season = np.random.choice(schedule.season.unique())
# schedule = schedule[schedule.season == season]
# week = np.random.choice(schedule.week.unique())
# schedule = schedule[schedule.week == week]
# home_team = np.random.choice(schedule.home_team.unique())
# schedule = schedule[schedule.home_team == home_team]
# away_team = schedule.away_team.unique()[0]
week = 20
season = 2022
coaches = coaches[(coaches.week == week) & (coaches.season == season)]
home_team = "PHI"
home_coach = coaches[coaches.team==home_team].coach.unique()[0]
away_team = "SF"
away_coach = coaches[coaches.team==away_team].coach.unique()[0]

home_team = Team(home_team, week, season, home_coach)
away_team = Team(away_team, week, season, away_coach)
start = time.time()
home_scores = []
away_scores = []
#%%
for i in range(1,101):
    if __name__=='__main__':
        try:       
            game = Game(home_team, away_team,i)
        except Exception as error:
            print(traceback.format_exc())
            continue
    home_scores.append(game.home_team_score)
    away_scores.append(game.away_team_score)
    print(time.time() - start)
