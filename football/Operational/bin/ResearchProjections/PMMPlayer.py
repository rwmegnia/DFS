#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 00:19:28 2022

@author: robertmegnia
"""
import pandas as pd
def getPMM(df,season,week):
    pos_frames=[]
    methods=['BR','EN','NN','RF','GB','Tweedie']
    df['ML']=df[methods].mean(axis=1)
    for pos in ['QB','RB','WR','TE','DST']:
        pos_frame=df[(df.position==pos)&(df.season==season)&(df.week==week)]
        members=pd.concat([pos_frame[m] for m in methods])
        members.sort_values(ascending=False,inplace=True)
        members.name='PMM'
        members=members.to_frame()
        pos_frame.sort_values(by='ML',ascending=False,inplace=True)
        members=members[len(methods)-1::len(methods)]
        pos_frame['PMM']=members.values
        pos_frames.append(pos_frame)
    df=pd.concat(pos_frames)
    df.reset_index(drop=True,inplace=True)
    df['ML']=df[['PMM']+methods].mean(axis=1)
    return df