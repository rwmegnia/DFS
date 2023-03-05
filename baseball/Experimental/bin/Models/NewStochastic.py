#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 03:31:31 2022

@author: robertmegnia
"""
def BattersStochastic(players,stats_df):
    # Make sure switch hitters have either bat hand opposite that of opposing pitcher hand
    players.loc[(players.handedness=='S')&(players.opp_pitcher_hand=='R'),'handedness']='L'
    players.loc[(players.handedness=='S')&(players.opp_pitcher_hand=='L'),'handedness']='R'
    players['splits']='vs_'+players.opp_pitcher_hand+'HP'
    
    # Filter Players to Batters and set index to ID and splits to join with stats frame
    batters=players[players.position_type!='Pitcher']
    batters.set_index(['player_id','splits'],inplace=True)
    stats_df.set_index(['player_id','splits'],inplace=True)
    batters=batters.join(stats_df[DK_Batter_Stats],lsuffix='_a')
    
    # Get each batters last 20 games
    batters=batters.groupby('player_id').tail(40)
    
    # Scale batter stats frame to z scores to get opponent influence
    # Get batter stats from last 40 games
    scaled_batters = stats_df[stats_df.game_date>'2022-01-1'].groupby(['player_id','splits']).tail(40)
    scaled_batters.sort_index(inplace=True)
    # Get batters average/standard dev stats over last 20 games
    scaled_averages = scaled_batters.groupby(['player_id','splits'],as_index=False).rolling(window=20,min_periods=5).mean()[DK_Batter_Stats].groupby(['player_id','splits']).tail(20)
    scaled_stds = scaled_batters.groupby(['player_id','splits'],as_index=False).rolling(window=20,min_periods=5).std()[DK_Batter_Stats].groupby(['player_id','splits']).tail(20)
    scaled_batters=scaled_batters.groupby(['player_id','splits']).tail(20)
    # Get batters Z scores over last 20 games
    scaled_batters[DK_Batter_Stats] = ((scaled_batters[DK_Batter_Stats]-scaled_averages[DK_Batter_Stats])/scaled_stds[DK_Batter_Stats]).values
    opp_pitcher_stats = scaled_batters.groupby(['opp_pitcher_id','game_date']).mean().groupby(['opp_pitcher_id']).tail(7)[DK_Batter_Stats].reset_index()
    opp_pitcher_stats = opp_pitcher_stats[opp_pitcher_stats.opp_pitcher_id.isin(players.opp_pitcher_id)].groupby('opp_pitcher_id').mean()
    quantiles=scipy.stats.norm.sf(opp_pitcher_stats)
    quantiles=pd.DataFrame(quantiles,columns=DK_Batter_Stats).set_index(opp_pitcher_stats.index)
    #%%
    averages=batters.groupby('player_id').mean()
    averages.opp_pitcher_id=averages.opp_pitcher_id.astype(int)
    averages=averages.reset_index().set_index('opp_pitcher_id')
    stds=batters.groupby('player_id').std()
    quantiles=averages.join(quantiles[DK_Batter_Stats],lsuffix='_quant')[DK_Batter_Stats]
    averages.sort_index(inplace=True)
    sims=np.random.normal(averages[DK_Batter_Stats],stds[DK_Batter_Stats],size=(10000,len(averages),len(DK_Batter_Stats)))
    sims[sims<0]=0
    #
    batters=players[players.position_type!='Pitcher']
    low=pd.DataFrame(np.quantile(sims,0.1,axis=0),columns=DK_Batter_Stats).set_index(averages.player_id)
    low.rename({'DKPts':'Floor1'},axis=1,inplace=True)
    low['Floor']=getDKPts(low,'batters')
    low['Floor']=low[['Floor','Floor1']].mean(axis=1)
    high=pd.DataFrame(np.quantile(sims,0.9,axis=0),columns=DK_Batter_Stats).set_index(averages.player_id)
    high['Ceiling']=getDKPts(high,'batters')
    high['Ceiling']=high[['Ceiling','DKPts']].mean(axis=1)
    median=pd.concat([pd.DataFrame(np.diag(pd.DataFrame(sims[:,i,:],columns=DK_Batter_Stats).quantile(quantiles.values[i,:])).reshape(1,-1),columns=DK_Batter_Stats) for i in range(0,len(batters))]).set_index(averages.player_id)
    median.rename({'DKPts':'Stochastic1'},axis=1,inplace=True)
    median['Stochastic']=getDKPts(median,'batters')
    median['Stochastic']=median[['Stochastic','Stochastic1']].mean(axis=1)
    batters.set_index('player_id',inplace=True)
    batters=batters.join(low['Floor'].round(1))
    batters=batters.join(high['Ceiling'].round(1))
    batters=batters.join(median['Stochastic'].round(1))
    return batters

def PitcherStochastic(players,stats_df):
    