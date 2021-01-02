import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
import altair as alt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

Player_URL = ('F:\Football_data\Players.csv')
Team_URL = ('F:\Football_data\Teams.csv')
Keeper_URL = ('F:\Football_data\Keepers.csv')

def main():
    player_df = load_data()
    team_df = load_data1()
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Stats Exploration", "Players Comparision", "Scout Watch"])
    seasons = ['2020/2021','2019/2020','2018/2019','2017/2018']
    leagues = ['Premier League', 'SerieA', 'La Liga', 'Bundesliga', 'Champions League']
    options = ['Players Comparision', 'Single Player Analysis']
    positions = ['FW','FW,MF','MF,FW','MF','MF,DF','DF,MF','DF','GK']
    df2 = team_df[team_df.columns.difference(['squad', 'Season','League'])]
    unique = player_df.player.unique()

    if page == "Homepage":
        st.header("This is an advance football data explorer.")
        st.write("Please select a page on the left for different analysis.")
        st.markdown("![Alt Text](https://i.pinimg.com/originals/71/64/5d/71645d4afa6ff297eb32868d0010c6be.gif)")

    elif page == "Stats Exploration":
        st.title("Advanced Teams Stats Exploration")

        col1,col2,col3= st.beta_columns(3)
        col4,col5,col6 = st.beta_columns(3)
        with col1:
        	league = st.selectbox("Choose a league", leagues)
        with col2:
        	season = st.selectbox("Choose a season", seasons)
        with col3:
        	attribute = st.selectbox("Choose an attribute", df2.columns)
        with col5:
        	submit = st.button('Submit')
        if submit:
        	visualize_teams_attribute(team_df, league, season, attribute)
        else:
        	visualize_teams(team_df,league,season)
    elif page == "Players Comparision":
        st.title("Players comparision and player analysis")
        sub = st.selectbox("Choose the option of comparision or single player analysis", options)
        col7,col8,col9 = st.beta_columns(3)
        col10,col11 = st.beta_columns(2)
        if sub == 'Players Comparision':
        	with col7:
        		player_1 = st.selectbox("Choose the first player", unique)
        	with col8:
        		player_2 = st.selectbox("Choose the second player", unique)
        	with col9:
        		player_season = st.selectbox("Choose the season", seasons)
        	visualize_players(player_df, player_1, player_2, player_season)
        else:
        	with col10:
        		player_1 = st.selectbox("Choose the player for analysis", unique)
        	with col11:
        		player_season = st.selectbox("Choose the season", seasons)
        	visualize_single_player(player_df,player_1, player_season)
    elif page == "Scout Watch":
    	st.title("Check out the best players for each position in the league each season")
    	col12,col13,col14 = st.beta_columns(3)
    	col15,col16,col17 = st.beta_columns(3)
    	with col12:
    		league = st.selectbox("Choose the league", leagues)
    	with col13:
    		position = st.selectbox("Choose the position", positions)
    	with col14:
    		season = st.selectbox("Choose the season", seasons)
    	with col16:
    		submit = st.button('Submit')
    	if submit:
    		show_top_players(player_df,league,position,season)


@st.cache
def load_data():
    df = pd.read_csv(Player_URL)
    return df

@st.cache
def load_data1():
    df = pd.read_csv(Team_URL)
    return df

@st.cache
def load_data2():
    df = pd.read_csv(Keeper_URL)
    return df


def visualize_teams(df, league, season):
	league_data = df[df['League'] == league]
	season_data = league_data[league_data['Season'] == season]
	fig = make_subplots(rows=2, cols=2,subplot_titles=("Goals scored", "Assists", "Goals per 90", "Assists per 90"))

	fig.add_trace(go.Bar(x=season_data['goals'].values,y=season_data.squad,
                    orientation='h',
                    name='Goals scored by and against Teams'),row=1,col=1)
	fig.add_trace(go.Bar(x=season_data['assists'].values,y=season_data.squad,
					orientation='h',
                    name='Assists by and against Teams'),row=1,col=2)
	fig.add_trace(go.Bar(x=season_data['goals_per90'].values,y=season_data.squad,
					orientation='h',
                    name='Goals per 90 minutes'),row=2,col=1)
	fig.add_trace(go.Bar(x=season_data['assists_per90'].values,y=season_data.squad,
					orientation='h',
                    name='Assists per 90 minutes'),row=2,col=2)

	fig.update_yaxes(
		ticks="outside",
		nticks=10,
		ticklen=5
		)
	fig.update_layout(yaxis={'categoryorder':'total ascending'},yaxis2={'categoryorder':'total ascending'},yaxis3={'categoryorder':'total ascending'},yaxis4={'categoryorder':'total ascending'},yaxis5={'categoryorder':'total ascending'})
	fig.update_layout(height=1000, width=1000)
	fig.update_layout(
    title={
        'text': "Basic attack and defence analysis of the current season",
        'y':0.98,
        'x':0.40,
        'xanchor': 'center',
        'yanchor': 'top'})

	st.plotly_chart(fig)


def visualize_teams_attribute(df, league, season, attribute):
	league_data = df[df['League'] == league]
	season_data = league_data[league_data['Season'] == season]
	fig = go.Figure(go.Bar(
				x = season_data[attribute].values,
				y = season_data.squad,
				orientation ='h'
		))
	fig.update_layout(yaxis={'categoryorder':'total ascending'})
	fig.update_layout(height=800, width=800)
	fig.update_layout(
    title={
        'text': 'Teams' + ' ' + str(attribute) + ' ' + 'stats of the' + ' ' + str(season) + ' ' + ' season',
        'y':0.98,
        'x':0.40,
        'xanchor': 'center',
        'yanchor': 'top'})
	st.plotly_chart(fig)


def visualize_players(df, player_1, player_2, season):
	season_data = df[df['Season'] == season]
	player_1_data = season_data[season_data['player'] == player_1]
	player_2_data = season_data[season_data['player'] == player_2] 
	st.write(player_1_data)
	st.write(player_2_data)
	player_1_data['position']= player_1_data['position'].astype(str)
	pos = player_1_data['position'].str.lower()
	position_match = pos.str.contains(str(player_2_data['position'].values).lower(),regex=True)
	if position_match.iloc[0] == 1:
		if ((player_1_data['position'].values == 'FW') or (player_1_data['position'].values == 'FW,MF') or (player_1_data['position'].values == 'MF,FW')):

			data = [go.Scatterpolar(
  				r = [player_1_data['goals_assists_per90'].values[0],player_1_data['goals_per_shot_on_target'].values[0],player_1_data['xg_xa_per90'].values[0],player_1_data['passes_into_final_third'].values[0],player_1_data['passes_into_penalty_area'].values[0],player_1_data['through_balls'].values[0],player_1_data["touches_att_pen_area"].values[0]],
  				theta = ['goals and assists per90','goals per shot on target','npxg per shot','passes into final third','passes into penalty area','through balls','touches attacking penalty area'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
                color = 'cyan'
            		)
        		),
    			go.Scatterpolar(
    				r = [player_2_data['goals_assists_per90'].values[0],player_2_data['goals_per_shot_on_target'].values[0],player_2_data['xg_xa_per90'].values[0],player_2_data['passes_into_final_third'].values[0],player_2_data['passes_into_penalty_area'].values[0],player_2_data['through_balls'].values[0],player_2_data["touches_att_pen_area"].values[0]],
  					theta = ['goals and assists per90','goals per shot on target','xg xa per 90','passes into final third','passes into penalty area','through balls','touches attacking penalty area'],
  					fill = 'toself',
  					name= str(player_2),
     				line =  dict(
                	color = 'orange'
            		)
				)]

			layout = go.Layout(
  				polar = dict(
    			radialaxis = dict(
      			visible = True,
      			range = [0, 200]
    				)
  				),
  				showlegend = True,
  				title = "{} vs {} Stats Comparison".format(str(player_1), str(player_2))
  				)
			fig = go.Figure(data=data, layout=layout)
			st.plotly_chart(fig)

		elif ((player_1_data['position'].values == 'MF') or (player_1_data['position'].values =='MF,DF')):
			data = [go.Scatterpolar(
  				r = [player_1_data['sca_per90'].values[0],player_1_data['gca_per90'].values[0],player_1_data['xg_xa_per90'].values[0],player_1_data['passes_into_final_third'].values[0],player_1_data['passes_into_penalty_area'].values[0],player_1_data['through_balls'].values[0],player_1_data["progressive_passes"].values[0]],
  				theta = ['shot creation per 90','goals creation per 90','xg xa per 90','passes into final third','passes into penalty area','through balls','progressive passes'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
            	color = 'orange'
        			)
				),
				go.Scatterpolar(
  				r = [player_2_data['sca_per90'].values[0],player_2_data['gca_per90'].values[0],player_2_data['xg_xa_per90'].values[0],player_2_data['passes_into_final_third'].values[0],player_2_data['passes_into_penalty_area'].values[0],player_2_data['through_balls'].values[0],player_2_data["progressive_passes"].values[0]],
  				theta = ['shot creation per 90','goals creation per 90','xg xa per 90','passes into final third','passes into penalty area','through balls','progressive passes'],
  				fill = 'toself',
  				name= str(player_2),
     			line =  dict(
            	color = 'orange'
        			)
				)]

			layout = go.Layout(
  				polar = dict(
    			radialaxis = dict(
      			visible = True,
      			range = [0, 200]
    			)
  			),
  				showlegend = True,
  				title = "{} vs {} Stats Comparison".format(str(player_1), str(player_2))
			)
			fig = go.Figure(data=data, layout=layout)
			st.plotly_chart(fig)

		elif ((player_1_data['position'].values == 'DF') or (player_1_data['position'].values == 'DF,MF')):
			data = [go.Scatterpolar(
  				r = [player_1_data['aerials_won_pct'].values[0],player_1_data['dispossessed'].values[0],player_1_data['interceptions'].values[0],player_1_data['tackles_won'].values[0],player_1_data['blocked_shots'].values[0],player_1_data['errors'].values[0],player_1_data["pens_conceded"].values[0]],
  				theta = ['aerial win percentage','Dispossessed','interceptions','Tackles won','blocked shots','errors','penalty conceded'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
            	color = 'orange'
        			)
				),
				go.Scatterpolar(
  				r = [player_2_data['aerials_won_pct'].values[0],player_2_data['dispossessed'].values[0],player_2_data['interceptions'].values[0],player_2_data['tackles_won'].values[0],player_2_data['blocked_shots'].values[0],player_2_data['errors'].values[0],player_2_data["pens_conceded"].values[0]],
  				theta = ['aerial win percentage','Dispossessed','interceptions','Tackles won','blocked shots','errors','penalty conceded'],
  				fill = 'toself',
  				name= str(player_2),
     			line =  dict(
            	color = 'orange'
        			)
				)]

			layout = go.Layout(
  				polar = dict(
    			radialaxis = dict(
      			visible = True,
      			range = [0, 200]
    				)
  				),
  				showlegend = True,
  				title = "{} vs {} Stats Comparison".format(str(player_1), str(player_2))
			)
			fig = go.Figure(data=data, layout=layout)
			st.plotly_chart(fig)

		elif (player_1_data['position'].values == 'GK'):
			data = [go.Scatterpolar(
  				r = [player_1_data['goals_against_per90_gk'].values[0],player_1_data['save_pct'].values[0],player_1_data['psnpxg_per_shot_on_target_against'].values[0],player_1_data['pens_saved'].values[0],player_1_data['cross_stopped_pct_gk'].values[0],player_1_data['errors'].values[0],player_1_data["miscontrols"].values[0]],
  				theta = ['Goals allowed per 90','Saves percentage','post shot xg saves on target','pens_saved','cross stopped pct','errors','miscontrols'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
            	color = 'orange'
        			)
				),
				go.Scatterpolar(
  				r = [player_2_data['goals_against_per90_gk'].values[0],player_2_data['save_pct'].values[0],player_2_data['psnpxg_per_shot_on_target_against'].values[0],player_2_data['pens_saved'].values[0],player_2_data['cross_stopped_pct_gk'].values[0],player_2_data['errors'].values[0],player_2_data["miscontrols"].values[0]],
  				theta = ['Goals allowed per 90','Saves percentage','post shot xg saves on target','pens_saved','cross stopped pct','errors','miscontrols'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
            	color = 'orange'
        			)
				)]

			layout = go.Layout(
  				polar = dict(
    			radialaxis = dict(
      			visible = True,
      			range = [0, 200]
    				)
  				),
  				showlegend = True,
  				title = "{} vs {} Stats Comparison".format(str(player_1), str(player_2))
				)
			fig = go.Figure(data=data, layout=layout)
			st.plotly_chart(fig)
			
	else:
		st.header('Please select players who plays in the same position')



def visualize_single_player(df, player_1, season):
	season_data = df[df['Season'] == season]
	player_1_data = season_data[season_data['player'] == player_1]
	st.write(player_1_data)
	if ((player_1_data['position'].values == 'FW') or (player_1_data['position'].values == 'FW,MF') or (player_1_data['position'].values == 'MF,FW')):

		data = [go.Scatterpolar(
  				r = [player_1_data['goals_assists_per90'].values[0],player_1_data['gca_per90'].values[0],player_1_data['xg_xa_per90'].values[0],player_1_data['passes_into_final_third'].values[0],player_1_data['passes_into_penalty_area'].values[0],player_1_data['through_balls'].values[0],player_1_data["touches_att_pen_area"].values[0]],
  				theta = ['shot on target pct','goals creation per 90','xg xa per 90','passes into final third','passes into penalty area','through balls','touches attacking penalty area'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
            	color = 'orange'
        		)
			)]

		layout = go.Layout(
  				polar = dict(
    			radialaxis = dict(
      			visible = True,
      			range = [0, 200]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution".format(player_1)
			)
		fig = go.Figure(data=data, layout=layout)
		st.plotly_chart(fig)
		
	elif ((player_1_data['position'].values == 'MF') or (player_1_data['position'].values =='MF,DF')):
		data = [go.Scatterpolar(
  				r = [player_1_data['sca_per90'].values[0],player_1_data['gca_per90'].values[0],player_1_data['xg_xa_per90'].values[0],player_1_data['passes_into_final_third'].values[0],player_1_data['passes_into_penalty_area'].values[0],player_1_data['through_balls'].values[0],player_1_data["progressive_passes"].values[0]],
  				theta = ['shot creation per 90','goals creation per 90','xg xa per 90','passes into final third','passes into penalty area','through balls','progressive passes'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
            	color = 'orange'
        		)
			)]

		layout = go.Layout(
  				polar = dict(
    			radialaxis = dict(
      			visible = True,
      			range = [0, 200]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution".format(player_1)
			)
		fig = go.Figure(data=data, layout=layout)
		st.plotly_chart(fig)
		


	elif ((player_1_data['position'].values == 'DF') or (player_1_data['position'].values == 'DF,MF')):
		data = [go.Scatterpolar(
  				r = [player_1_data['aerials_won_pct'].values[0],player_1_data['dispossessed'].values[0],player_1_data['interceptions'].values[0],player_1_data['tackles_won'].values[0],player_1_data['blocked_shots'].values[0],player_1_data['errors'].values[0],player_1_data["pens_conceded"].values[0]],
  				theta = ['aerial win percentage','Dispossessed','interceptions','Tackles won','blocked shots','errors','penalty conceded'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
            	color = 'orange'
        		)
			)]

		layout = go.Layout(
  				polar = dict(
    			radialaxis = dict(
      			visible = True,
      			range = [0, 200]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution".format(player_1)
			)
		fig = go.Figure(data=data, layout=layout)
		st.plotly_chart(fig)
		

	elif (player_1_data['position'].values == 'GK'):
		data = [go.Scatterpolar(
  				r = [player_1_data['fouls'].values[0],player_1_data['dispossessed'].values[0],player_1_data['ball_recoveries'].values[0],player_1_data['pens_conceded'].values[0],player_1_data['clearances'].values[0],player_1_data['errors'].values[0],player_1_data["miscontrols"].values[0]],
  				theta = ['Fouls','Dispossessed','ball recoveries','pens_conceded','cross stopped pct','errors','miscontrols'],
  				fill = 'toself',
  				name= str(player_1),
     			line =  dict(
            	color = 'orange'
        		)
			)]

		layout = go.Layout(
  				polar = dict(
    			radialaxis = dict(
      			visible = True,
      			range = [0, 200]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution".format(player_1)
			)
		fig = go.Figure(data=data, layout=layout)
		st.plotly_chart(fig)

def show_top_players(df,league,position,season):
	season_data = df[df['Season'] == season]
	position_data = season_data[season_data['position'] == position]	
	league_data = position_data[position_data['League'] == league]
	if ((position == 'FW') or (position == 'FW,MF') or (position == 'MF,FW')):
		league_data['per_index'] = league_data['goals_assists_per90'] + league_data['gca_per90'] + league_data['xg_xa_per90'] + league_data['passes_into_final_third'] + league_data['passes_into_penalty_area'] + league_data['through_balls'] + league_data['touches_att_pen_area']
		league_data.sort_values(by=['per_index'], ascending=False,inplace=True)
		df1 = league_data[['player', 'position','squad']]
		df1.reset_index(drop=True,inplace=True)
		st.write(df1)
	elif ((position == 'MF') or (position == 'MF,DF')):
		league_data['per_index'] = league_data['sca_per90'] + league_data['gca_per90'] + league_data['xg_xa_per90'] + league_data['passes_into_final_third'] + league_data['passes_into_penalty_area'] + league_data['through_balls'] + league_data['progressive_passes']
		league_data.sort_values(by=['per_index'], ascending=False,inplace=True)
		df1 = league_data[['player', 'position','squad']]
		df1.reset_index(drop=True,inplace=True)
		st.write(df1)
	elif ((position == 'DF') or (position == 'DF,MF')):
		league_data['per_index'] = league_data['aerials_won_pct'] - league_data['dispossessed'] + league_data['ball_recoveries'] - league_data['pens_conceded'] + league_data['clearances'] - league_data['errors'] - league_data['miscontrols']
		league_data.sort_values(by=['per_index'], ascending=False,inplace=True)
		df1 = league_data[['player', 'position','squad']]
		df1.reset_index(drop=True,inplace=True)
		st.write(df1)
	elif position == 'GK':
		league_data['per_index'] = league_data['ball_recoveries'] - league_data['dispossessed'] - league_data['fouls'] + league_data['pens_conceded'] + league_data['clearances'] - league_data['errors'] - league_data['miscontrols']
		league_data.sort_values(by=['per_index'], ascending=False,inplace=True)
		df1 = league_data[['player', 'position','squad']]
		df1.reset_index(drop=True,inplace=True)
		st.write(df1)

if __name__ == "__main__":
    main()





