#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:49:01 2023

@author: robertmegnia
"""
from SimulationModels import *
import numpy as np
class kickoff:
    def __init__(self,game):
        self.game=game
        if self.game.half_seconds_remaining == 1800:
            kickoff_from = 65
        else:
            kickoff_from = 100 - KickoffFrom_model.predict([game.qtr])[0]
        if (self.game.half_seconds_remaining == 1800) & (self.game.half == 2):
            if self.game.posteam != self.game.second_half_receiving_team:
                self.game.change_possession()
        self.game.yardline = kickoff_from
        kickoff_result = KickoffResult_model.predict([game.qtr])[0]
        print(
            f"{game.defteam.team_abbrev} kicks from their own {kickoff_from} yardline"
        )
        if kickoff_result == 'out_of_bounds':
            print("kickoff is out of bounds")
            self.game = self.kickoff_out_of_bounds()
        elif kickoff_result == 'touchback':
            self.game = self.kickoff_touchback()
        else:
            self.game = self.kickoff_return(kickoff_from)
            
    def kickoff_out_of_bounds(self,kickoff_result):
        self.game.yardline = 40
        return game
        
    def kickoff_touchback(self):
        self.game.yardline = TouchbackEndYardLine_model.predict([self.game.qtr])[0]
        if self.game.yardline < 25:
            penalty_yards = 25 - self.game.yardline
            print(
                f"Kickoff ends with a touchback. Penalty on {self.game.posteam.team_abbrev} for {penalty_yards} yards"
            )
            print(
                f"{self.game.posteam.team_abbrev} starts drive at {self.game.yardline}"
            )
        elif self.game.yardline > 25:
            penalty_yards = self.game.yardline - 25
            print(
                f"Kickoff ends with a touchback. Penalty on {self.game.defteam.team_abbrev} for {penalty_yards} yards"
            )
            print(
                f"{self.game.posteam.team_abbrev} starts drive at {self.game.yardline}"
            )
            return
        else:
            print("Kickoff ends with a touchback.")
            print(
                f"{self.game.posteam.team_abbrev} starts drive at {self.game.yardline}"
            )
        return self.game
    
    def kickoff_return(self,kickoff_from):
        kick_distance = KickoffDistance_model.predict([self.game.qtr])[0]
        returned_from = kickoff_from - kick_distance
        if returned_from < -9:
            returned_from = -9
        return_yards = KickoffReturn_model.predict([self.game.qtr])[0]
        # Make sure return gets out of endzone
        if (returned_from + return_yards) <= 0:
            return_yards = np.abs(returned_from) + 5
        print(
            f"{self.game.defteam.team_abbrev} kicks to {self.game.posteam.team_abbrev} {returned_from} yard line"
        )
        penalty = KickoffReturnedPenalty_model.predict([self.game.qtr])[0]
        fumble_lost = KickoffFumbleLost_model.predict([self.game.qtr])[0]
        fumble_td = KickoffFumbleLostReturnTD_model.predict(
                [self.game.qtr]
            )[0]
        if (returned_from + return_yards) >= 100:
            self.game=self.kickoff_return_td()
            return self.game
        
        elif penalty == True:
            print(
                f"{self.game.posteam.team_abbrev} returns ball {return_yards} yards to {returned_from+return_yards} yardline"
            )
            penalty_description = (
                KickoffReturnedPenaltyDefteam_model.predict([self.game.qtr])[0]
            )
            penalty_type = " ".join(penalty_description.split(" ")[:-1])
            penalty_yards = float(penalty_description.split(" ")[-1])
            self.game.yardline = returned_from + return_yards + penalty_yards
            print(
                f"{self.game.defteam.team_abbrev} comitted penalty {penalty_type}. {penalty_yards} yards added to the end of the play"
            )
            print(
                f"{self.game.posteam.team_abbrev} starts drive at {self.game.yardline} yardline"
            )
            elapsed_time = KickoffReturn_elapsed_time_model.predict(
                np.array(
                    [[return_yards, np.random.choice(range(0, 10000))]]
                )
            )[0]
            self.game.elapsed_time(elapsed_time)
            return self.game
        
        elif fumble_lost == 1:
            if fumble_td == True:
                print(
                    f"{self.game.posteam.team_abbrev} fumbles on play. Fumble recovered by {self.game.defteam.team_abbrev} for a touchdown!"
                )
                self.game.score_change(self.game.defteam, 6)
                self.game.change_possession()
                self.game.point_after_attempt(3)

            else:
                fumble_yards = (
                    KickoffFumbleLostReturnYards_model.predict(
                        [self.game.qtr]
                    )[0]
                )
                if (
                    fumble_yards
                    + (100 - (returned_from + return_yards))
                ) > 100:
                    fumble_yards = (returned_from + return_yards) - 1
                self.game.yardline = 100 - (
                    returned_from + return_yards - fumble_yards
                )
                print(
                    f"{self.game.posteam.team_abbrev} fumbles on play. Fumble recovered by {self.game.defteam.team_abbrev} and returned {fumble_yards} yards to {self.game.yardline} yardline"
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
                self.game.change_possession()
                self.game.elapsed_time(elapsed_time)
        else:
            self.game.yardline = returned_from = return_yards
            elapsed_time = KickoffReturn_elapsed_time_model.predict(
                np.array(
                    [[return_yards, np.random.choice(range(0, 10000))]]
                )
            )[0]
            self.game.elapsed_time(elapsed_time)
        return self.game
    def kickoff_return_td(self,returned_from):
        return_yards = 100 - returned_from
        print(
            f"{self.game.posteam.team_abbrev} returns ball {return_yards} yards for a touchdown!"
        )
        elapsed_time = KickoffReturnTD_elapsed_time_model.predict(
            np.array(
                [[return_yards, np.random.choice(range(0, 10000))]]
                )
        )[0]
        self.game.score_change(self.game.posteam, 6)
        self.game.point_after_attempt(elapsed_time)
        return self.game