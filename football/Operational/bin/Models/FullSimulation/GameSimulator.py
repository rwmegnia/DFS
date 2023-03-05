#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:24:48 2022

@author: robertmegnia
"""

import pandas as pd
import numpy as np
import re
from SimulationModels import *
import time
import nfl_data_py as nfl
from typing import Iterable, Callable, Dict, List, Tuple, TYPE_CHECKING
basedir = os.path.dirname(os.path.abspath(__file__))
model_dir=f'{basedir}/ml_models'
coaches=pd.read_csv('./coaches/coaching_logs.csv')

def getClass(classes,probs):
    classes=classes[probs>0]
    probs=probs[probs>0]
    random=np.random.choice(np.arange(0.01,1.01,0.01))
    probs=probs.cumsum()-random
    try:
        idx=list(probs).index(min([i for i in list(probs) if i >=0]))
    except ValueError:
        idx=np.where(probs==probs.max())[0][0]
    return classes[idx]

class Player(pd.DataFrame):
    def __init__(self,player):
        super().__init__(player)
        self['depth_team'] = player.depth_team.values[0]
        self['first_name'] = player.first_name.values[0]
        self['last_name'] = player.last_name.values[0]
        self['football_name'] = player.football_name.values[0]
        self['formation'] = player.formation.values[0]
        self['gsis_id'] = player.gsis_id.values[0]
        self['jersey_number'] = player.jersey_number.values[0]
        self['position'] = player.position.values[0]
        self['depth_position'] = player.depth_position.values[0]
        self['full_name'] = player.full_name.values[0]
        self.name = self.full_name.values[0]
        self.pos = self.position.values[0]
    def __repr__(self):
        return f'{self.name} - {self.pos}'
    
    
    
    
    
        
class Team(object):
    def __init__(self,team,week,season):
        self.coach=coaches[coaches.team==team].coach.unique()[0]
        self.coach_string=''.join(self.coach.split(' '))
        self.team_abbrev=team
        self.import_coach_models()
        self.depth_chart = self.import_roster(week,season)
    
    def __repr__(self):
        return self.team_abbrev
    
    def import_coach_models(self):
        self.RunPass_model=pickle.load(open(f'{model_dir}/CoachingModels/{self.coach_string}_RunPass_model.pkl','rb'))
        self.RunFormation_model=pickle.load(open(f'{model_dir}/CoachingModels/{self.coach_string}_RunFormation_model.pkl','rb'))
        self.PassFormation_model=pickle.load(open(f'{model_dir}/CoachingModels/{self.coach_string}_PassFormation_model.pkl','rb'))
        self.Personnel_models={'Run':{},'Pass':{}}
        for model in os.listdir(f'{model_dir}/CoachingModels'):
            if (f'{self.coach_string}_Run' in model)&('_personnel' in model):
                formation=re.findall(f'{self.coach_string}_Run([A-Z].+)_personnel',model)[0]                
                self.Personnel_models['Run'][formation]=pickle.load(open(f'{model_dir}/CoachingModels/{self.coach_string}_Run{formation}_personnel_model.pkl','rb'))
            elif (f'{self.coach_string}_Pass' in model)&('_personnel' in model):
                formation=re.findall(f'{self.coach_string}_Pass([A-Z].+)_personnel',model)[0]                
                self.Personnel_models['Pass'][formation]=pickle.load(open(f'{model_dir}/CoachingModels/{self.coach_string}_Run{formation}_personnel_model.pkl','rb'))
        return
    
    def import_roster(self,week,season):
        roster=nfl.import_depth_charts([season])
        roster['depth_team']=roster.depth_team.astype(float)
        roster=roster[(roster.season==season)&
                                (roster.week==week)&
                                (roster.club_code==self.team_abbrev)]
        depth_chart={}
        for player in roster.iterrows():
            depth_position=player[1]['depth_position']
            gsis_id=player[1]['gsis_id']
            player = roster[roster.gsis_id==gsis_id]
            if depth_position in depth_chart.keys():
                depth_chart[depth_position].append(Player(player))
            else:
                depth_chart[depth_position]=[Player(player)]
        return depth_chart
    
    def depth_chart(self):
        depth_chart = {}
        for player in self.roster:
            
            depth_chart[f'{player.depth_position}{player.depth_team}']=player
        
    
class Game(object):
    def __init__(self,home_team,away_team):
        self.home_team=home_team
        self.away_team=away_team
        self.home_timeouts_remaining=3
        self.away_timeouts_remaining=3
        self.home_team_score = 0
        self.away_team_score = 0
        self.posteam_score=0
        self.away_team_score=0
        self.qtr = 1
        self.half = 1
        self.game_seconds_remaining = 3600
        self.half_seconds_remaining = 1800
        self.qtr_seconds_remaining = 900
        self.play_clock=40
        self.end_game=False
        self.overTime=False
        self.OT_first_possession=True
        self.start_simulation()
        
    def start_simulation(self):
        self.posteam,self.defteam = self.coinToss()
        self.kickoff()
        
    def run_play(self):
        play_results=Play(self)
        return
    
    def coinToss(self):
        guess = np.random.choice(['heads','tails'])
        result = np.random.choice(['heads','tails'])
        print('Opening Coin Toss')
        print(f'{self.away_team.team_abbrev} calls {guess}')
        print(f'Result is {result}')
        decision = np.random.choice(['kick','receive'])
        if self.overTime==True:
            decision='receive'
        if guess==result:
            print(f'{self.away_team.team_abbrev} wins coin toss and elects to {decision}')
            if decision=='kick':
                self.second_half_receiving_team = self.away_team
                self.posteam_score=self.home_team_score
                self.defteam_score=self.away_team_score
                return self.home_team,self.away_team
            else:
                self.second_half_receiving_team = self.home_team
                self.posteam_score=self.away_team_score
                self.defteam_score=self.home_team_score
                return self.away_team,self.home_team
        else:
            print(f'{self.home_team.team_abbrev} wins coin toss and elects to {decision}')
            if decision=='kick':
                self.second_half_receiving_team = self.home_team                
                self.second_half_kicking_team = self.away_team
                self.posteam_score=self.away_team_score
                self.defteam_score=self.home_team_score
                return self.away_team,self.home_team
            else:
                self.second_half_receiving_team = self.away_team
                self.second_half_kicking_team = self.home_team
                self.posteam_score=self.home_team_score
                self.defteam_score=self.away_team_score
                return self.home_team,self.away_team
            
    def kickoff(self):
        if self.half_seconds_remaining==1800:
            kickoff_from = 65
        else:
            kickoff_from = 100-KickoffFrom_model.predict([self.qtr])[0]
        if (self.half_seconds_remaining==1800)&(self.half==2):
            if self.posteam!=self.second_half_receiving_team:
                self.change_possession()
        self.yardline=kickoff_from
        kickoff_result = KickoffResult_model.predict([self.qtr])[0]
        print(f'{self.defteam.team_abbrev} kicks from their own {kickoff_from} yardline')
        if kickoff_result=='out_of_bounds':
            print('kickoff is out of bounds')
            self.yardline=40
            self.Drive=Drive(self)
            return
        elif kickoff_result=='touchback':
            self.yardline=TouchbackEndYardLine_model.predict([self.qtr])[0]
            if self.yardline<25:
                penalty_yards=25-self.yardline
                print(f'Kickoff ends with a touchback. Penalty on {self.posteam.team_abbrev} for {penalty_yards} yards')
                print(f'{self.posteam.team_abbrev} starts drive at {self.yardline}')
                self.Drive=Drive(self)
                return
            elif self.yardline>25:
                penalty_yards=self.yardline-25
                print(f'Kickoff ends with a touchback. Penalty on {self.defteam.team_abbrev} for {penalty_yards} yards')
                print(f'{self.posteam.team_abbrev} starts drive at {self.yardline}')
                self.Drive=Drive(self)
                return
            else:
                print(f'Kickoff ends with a touchback.')
                print(f'{self.posteam.team_abbrev} starts drive at {self.yardline}')
                self.Drive=Drive(self)
                return
        else:
            kick_distance = KickoffDistance_model.predict([self.qtr])[0]
            returned_from = kickoff_from-kick_distance
            # Ensure no returns from more than 5 yards deep in endzone
            if returned_from <-9:
                returned_from = -9
            return_yards = KickoffReturn_model.predict([self.qtr])[0]
            
            # Make sure return gets out of endzone
            if (returned_from+return_yards)<=0:
                return_yards = np.abs(returned_from)+5
            print(f'{self.defteam.team_abbrev} kicks to {self.posteam.team_abbrev} {returned_from} yard line')
            penalty = KickoffReturnedPenalty_model.predict([self.qtr])[0]
            if (returned_from+return_yards) >=100:
                return_yards=100-returned_from
                print(f'{self.posteam.team_abbrev} returns ball {return_yards} yards for a touchdown!')
                elapsed_time=KickoffReturnTD_elapsed_time_model.predict(np.array([[return_yards,np.random.choice(range(0,10000))]]))[0]
                self.score_change(self.posteam, 6)
                self.point_after_attempt(elapsed_time)
                return
            print(f'{self.posteam.team_abbrev} returns ball {return_yards} yards to {returned_from+return_yards} yardline')
            if penalty==True:
                penalty_description=KickoffReturnedPenaltyDefteam_model.predict([self.qtr])[0]
                penalty_type=' '.join(penalty_description.split(' ')[:-1])
                penalty_yards=float(penalty_description.split(' ')[-1])
                self.yardline=returned_from+return_yards+penalty_yards
                print(f'{self.defteam.team_abbrev} comitted penalty {penalty_type}. {penalty_yards} yards added to the end of the play')
                print(f'{self.posteam.team_abbrev} starts drive at {self.yardline} yardline')
                elapsed_time=KickoffReturn_elapsed_time_model.predict(np.array([[return_yards,np.random.choice(range(0,10000))]]))[0]
                self.elapsed_time(elapsed_time)
                self.Drive=Drive(self)
                return
            fumble = KickoffFumble_model.predict([self.qtr])[0]
            if fumble==1:
                fumble_lost = KickoffFumbleLost_model.predict([self.qtr])[0]
                if fumble_lost==1:
                    fumble_td = KickoffFumbleLostReturnTD_model.predict([self.qtr])[0]
                    if fumble_td ==True:
                        print(f'{self.posteam.team_abbrev} fumbles on play. Fumble recovered by {self.defteam.team_abbrev} for a touchdown!')
                        self.score_change(self.defteam,6)
                        self.change_possession()
                        self.point_after_attempt(3)

                    else:
                        fumble_yards = KickoffFumbleLostReturnYards_model.predict([self.qtr])[0]
                        if (fumble_yards +(100-(returned_from+return_yards))) > 100:
                            fumble_yards = (returned_from+return_yards)-1                          
                        self.yardline = 100-(returned_from+return_yards-fumble_yards)
                        print(f'{self.posteam.team_abbrev} fumbles on play. Fumble recovered by {self.defteam.team_abbrev} and returned {fumble_yards} yards to {self.yardline} yardline')
                        print(return_yards,fumble_yards)
                        elapsed_time=KickoffReturnFumble_elapsed_time_model.predict(np.array([[return_yards,fumble_yards,np.random.choice(range(0,10000))]]))[0]
                        self.change_possession()
                        self.elapsed_time(elapsed_time)
                        self.Drive=Drive(self)
                else:
                    self.yardline=returned_from+return_yards
                    print(f'{self.posteam.team_abbrev} fumbles on play but recovers at {self.yardline} yardline')
                    elapsed_time=KickoffReturn_elapsed_time_model.predict(np.array([[return_yards,np.random.choice(range(0,10000))]]))[0]
                    self.elapsed_time(elapsed_time)
                    self.Drive=Drive(self)
            else:
                self.yardline=returned_from=return_yards
                elapsed_time=KickoffReturn_elapsed_time_model.predict(np.array([[return_yards,np.random.choice(range(0,10000))]]))[0]
                self.elapsed_time(elapsed_time)    
                self.Drive=Drive(self)
      
    def point_after_attempt(self,elapsed_time):
        if self.end_game==True:
            return
        print('Extra Point Attempt')
        two_points = self.two_point_attempt_decision()
        two_points=False
        if two_points==True:
            print(f'{self.posteam.team_abbrev} Attempts 2 Point Conversion')
            self.run_two_point_conversion()
        else:
            self.extra_point()
        self.elapsed_time(elapsed_time)
        self.change_possession()
        if self.end_game!=True:
            self.kickoff()
        
    
    def score_change(self,scoring_team,points):
        if scoring_team==self.home_team:
            self.home_team_score+=points
        else:
            self.away_team_score+=points
        if (self.overTime==True)&(points in [2,6]):
            self.end_game=True
            self.elapsed_time(600)
        if (self.overTime==True)&(self.OT_first_possession==False):
            if self.home_team_score!=self.away_team_score:
                self.end_game=True
                self.elapsed_time(600)

            
    def change_possession(self):
        if self.home_team==self.posteam:
            self.posteam=self.away_team
            self.posteam_score=self.away_team_score
            self.defteam=self.home_team    
            self.defteam_score=self.home_team_score
        else:
            self.posteam=self.home_team
            self.posteam_score=self.home_team_score
            self.defteam=self.away_team
            self.defteam_score=self.away_team_score
        if self.overTime==True:
            self.OT_first_possession=False
    
    def elapsed_time(self,time):
        self.game_seconds_remaining-=time
        self.half_seconds_remaining-=time
        if self.half_seconds_remaining<=0:
            print(f'End of Half {self.half}')
            self.half+=1
            self.half_seconds_remaining=1800
        self.qtr_seconds_remaining-=time
        if self.qtr_seconds_remaining<=0:
            print(f'End of Quarter {self.qtr}')
            self.qtr+=1
            self.qtr_seconds_remaining=900
        if self.game_seconds_remaining<=0:
            if self.qtr==4: 
                self.home_team_score=self.away_team_score
            if self.home_team_score==self.away_team_score:
                if self.qtr==4:
                    print('Going into overtime!')
                    self.game_seconds_remaining=600
                    self.qtr_seconds_remaining=600
                    self.half_seconds_remaining=600
                    self.qtr=5
                    self.overTime=True
                    self.coinToss()
                    self.kickoff()
            else:
                self.end_game=True
                if self.home_team_score>self.away_team_score:
                    winner=self.home_team.team_abbrev
                    winning_score = self.home_team_score
                    losing_score = self.away_team_score
                else:
                    winner=self.away_team.team_abbrev
                    winning_score = self.away_team_score
                    losing_score = self.home_team_score
                if self.overTime==True:
                    print(f'{winner} wins {winning_score} - {losing_score} in over time')
                else:
                    print(f'{winner} wins {winning_score} - {losing_score} in over time')
                return


    
    def two_point_attempt_decision(self):
        if self.qtr<4:
            prob_no=TwoPointDecision_model.predict_proba(np.array([[self.posteam_score-self.defteam_score,self.game_seconds_remaining]]))[0][0]
            random_choice = np.random.choice(np.arange(0,1,0.01))
            if random_choice < prob_no:
                return False
            else:
                return True
        else:
            return TwoPointDecision_model.predict(np.array([[self.posteam_score-self.defteam_score,self.game_seconds_remaining]]))[0]
    
    def extra_point(self):
        kick_from=XPFrom_model.predict([self.qtr])[0]
        kick_result_probs=XP_model.predict_proba([[kick_from]]).round(2)[0]
        random=np.random.choice(np.arange(0,1,0.01))
        if random<kick_result_probs[0]:
            kick_result='blocked'
            safety=XPBlockedReturn_model.predict([self.qtr])[0]
            if safety==True:
                self.score_change(self.defteam,2)  
        elif (random>=kick_result_probs[0])&(random<kick_result_probs[1]):
            kick_result='failed'
        else:
            kick_result='good'
            self.score_change(self.posteam, 1)
        print(f'{self.posteam.team_abbrev} kicks extra point from {kick_from} yardline. Extra point attempt {kick_result}')

class Drive(Game):
    def __init__(self,Game):
        self.Game=Game
        self.down=1
        self.ydstogo=10
        self.play_clock=40
        self.clock_running = False
        self.qtr=Game.qtr
        self.yardline=Game.yardline
        self.half_seconds_remaining=Game.half_seconds_remaining
        self.game_seconds_remaining=Game.game_seconds_remaining
        self.score_differential=Game.posteam_score-Game.defteam_score
        self.start_drive()
        
    def start_drive(self):
        play_type_probs = self.Game.posteam.RunPass_model.predict_proba([[self.qtr,
                                                         self.down,
                                                         self.ydstogo,
                                                         self.yardline,
                                                         self.half_seconds_remaining,
                                                         self.game_seconds_remaining,
                                                         self.score_differential]])[0].round(2)
        self.play_type=getClass(np.array(['pass','run']),play_type_probs)
        if self.play_type=='pass':
            play_type='Pass'
            formations=self.Game.posteam.PassFormation_model.classes_
            formation_probs = self.Game.posteam.PassFormation_model.predict_proba([[self.down,
                                                                         self.ydstogo,
                                                                             self.half_seconds_remaining,
                                                                             self.score_differential]])[0].round(2)
            self.formation = getClass(formations,formation_probs) 
        else:
            play_type='Run'
            formations=self.Game.posteam.PassFormation_model.classes_
            formation_probs = self.Game.posteam.RunFormation_model.predict_proba([[self.down,
                                                                             self.ydstogo,
                                                                             self.half_seconds_remaining,
                                                                             self.score_differential]])[0].round(2)
            self.formation=getClass(formations,formation_probs)
            
        self.personnel_model = self.Game.posteam.Personnel_models[play_type][self.formation]
        self.personnel_types=self.personnel_model.classes_
        self.personnel_probs=self.personnel_model.predict_proba([[self.down,
                                                        self.ydstogo]])[0].round(2)
        self.personnel=getClass(self.personnel_types, self.personnel_probs)
        self.run_play(self.play_type,self.formation, self.personnel)
        return  

    def run_play(self,play_type,formation,personnel):
        print(f'{self.Game.posteam.team_abbrev} lines up in {formation} formation with {personnel}')
        self.lineup_players(personnel)
        #self.pre_snap_penalty()
    
    def lineup_players(self,personnel):
        return
#%%
week=2
season=2021
coaches=coaches[(coaches.week==week)&(coaches.season==season)]
home_team=Team('PIT',week,season)
away_team=Team('NE',week,season)
start=time.time()
home_scores=[]
away_scores=[]
game=Game(home_team,away_team)
home_scores.append(game.home_team_score)
away_scores.append(game.away_team_score)
print(time.time()-start)