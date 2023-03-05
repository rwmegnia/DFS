#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:06:27 2023

@author: robertmegnia
"""

def messageHandler(self,message_type,*args):
    print(message_type)
    if message_type == 'Opening Coin Toss':
        away_team=args[0]
        home_team=args[1]
        guess=args[2]
        result=args[3]
        decision=args[4]
        print("Opening Coin Toss")
        print(f"{self.away_team.team_abbrev} calls {guess}")
        print(f"Result is {result}")
        if guess==result:
            print(
                f"{self.away_team.team_abbrev} wins coin toss and elects to {decision}"
            )
        else:
            print(
                f"{self.home_team.team_abbrev} wins coin toss and elects to {decision}"
            )
    elif message_type == 'punt_block_return_td':
        posteam=args[0]
        defteam=args[1]
        time = args[2]
        print(f'{posteam} punt is blocked. {defteam} recovers and returns for a touchdown! ({time})')
    
    elif message_type =='punt_block_safety':
        posteam=args[0]
        defteam=args[1]
        time=args[2]
        print(f'{posteam} punt is blocked!. Block results in a safety! ({time})')
    
    elif message_type =='punt_block_ob':
        posteam=args[0]
        defteam=args[1]
        ob_at =args[2]
        time=args[3]
        print(f'{posteam} punt is blocked!. Ball goes out of bounds at {ob_at} yardline. ({time})')
    
    elif message_type == 'punt_block_return':
        posteam = args[0]
        defteam = args[1]
        recovered_by=args[2]
        if recovered_by==0:
            recovered_by=defteam
        else:
            recovered_by=posteam
        recovered_at=args[3]
        return_yards=args[4]
        yardline=args[5]
        print(f'{posteam} punt is blocked!. {recovered_by} recovers at {recovered_at} yardline and returns for {return_yards} yards to {yardline}!. ({time})')

    elif message_type == 'punt_touchback':
        posteam = args[0]
        defteam = args[1]
        time = args[2]
        print(f'{posteam} punts into the endzone. Result is a touchback. ({time})')
    
    elif message_type =='punt_ob':
        posteam = args[0]
        defteam = args[1]
        distance = args[2]
        result = args[3]
        yardline = args[4]
        time = args[5]
        print(f'{posteam} punts {distance} yards to {yardline} yardline. Result is {result} ({time})')
    
    elif message_type == 'muffed_punt_td':
        posteam = args[0]
        defteam = args[1]
        time = args[2]
        print(f'{posteam} punts to {defteam}. {defteam} muffs punt! {posteam} recovers ball and returns for a touchdown!')
    
    elif message_type == 'muffed_punt_return':
        posteam = args[0]
        defteam = args[1]
        return_yards =args[2]
        yardline = args[3]
        time=args[4]
        print(f'{posteam} punts to {defteam}. {defteam} muffs punt! {posteam} recovers ball and returns {return_yards} yards to {yardline} yardline! ({time})')
    
    elif message_type == 'punt_return_td':
        posteam = args[0]
        defteam = args[1]
        return_yards = args[2]
        time = args[3]
        print(f'{posteam} punts to {defteam}. {defteam} returns {return_yards} yards for a touchdown! ({time})')

    elif message_type == 'punt_return':
        posteam = args[0]
        defteam = args[1]
        return_yards = args[2]
        yardline = args[3]
        time = args[4]
        print(f'{posteam} punts to {defteam}. {defteam} returns {return_yards} yards to {yardline} yardline. ({time})')
    
    elif message_type =='strip_sack_td':
        posteam = args[0]
        defteam = args[1]
        time = args[2]
        print(f'{posteam} QB sacked. Fumble recovered by {defteam} in end zone for a touchdown! ({time})')