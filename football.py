
import streamlit as st


# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import altair as alt
from scipy import stats
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


Player_URL = ('s3://footbalstat/Players.csv')
Team_URL = ('s3://footbalstat/Teams.csv')
Keeper_URL = ('s3://footbalstat/Keepers.csv')

hide_streamlit_style = """
					<style>
					#MainMenu {visibility: hidden;}
					footer {visibility: hidden;}
					</style>
					"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def main():
    player_df = load_data()
    team_df = load_data1()
    keeper_df = load_data2()
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Stats Exploration", "Players Comparision", "Scout Watch"])
    seasons = ['2020/2021','2019/2020','2018/2019','2017/2018']
    leagues = ['Premier League', 'SerieA', 'La Liga', 'Bundesliga', 'Ligue1']
    merged_df = pd.merge(team_df, player_df,  how='left', on=['League','Season','squad'])
    merged_df = merged_df[merged_df['player'].notna()]
    merged_df.drop_duplicates(inplace=True)
    analysis = ['Team analysis', 'Team and players analysis', 'Top players with choosen attributes']
    play_type = ['Attacking','Passing','Defending']
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
        option = st.selectbox('Choose the type of analysis', analysis )
        
        if option == 'Team analysis':

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

        elif option == 'Team and players analysis':
        	st.header('Choose the player for his contribution to the team stats')
        	col1,col2,col3= st.beta_columns(3)
        	col4,col5,col6 = st.beta_columns(3)
        	
        	with col1:
        		player = st.selectbox("Type in to choose a player", unique)
        	with col2:
        		season = st.selectbox('Choose a season', seasons)
        	with col3:
        		play = st.selectbox("Choose the type of play", play_type)
        	with col5:
        		submit = st.button('Submit')
        	if submit:
        		visualize_teams_players(merged_df,season,player,play)
        	else:
        		with col5:
        			st.markdown("![Alt Text](https://media.giphy.com/media/r8JxWQyDmE3C/source.gif)")

        elif option == 'Top players with choosen attributes':
        	st.header('Choose the attributes for the top players')
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
        		visualize_attribute_players(player_df, keeper_df, league, season, attribute)
        	else:
        		with col5:
        			st.markdown("![Alt Text](https://media.giphy.com/media/r8JxWQyDmE3C/source.gif)")

        else:
        	st.markdown("![Alt Text](https://media.giphy.com/media/r8JxWQyDmE3C/source.gif)")


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
        	visualize_players(player_df, keeper_df, player_1, player_2, player_season)
        elif sub == 'Single Player Analysis':
        	with col10:
        		player_1 = st.selectbox("Choose the player for analysis", unique)
        	with col11:
        		player_season = st.selectbox("Choose the season", seasons)
        	visualize_single_player(player_df, keeper_df ,player_1, player_season)
        else:
        	st.markdown("![Alt Text](https://i.pinimg.com/originals/71/64/5d/71645d4afa6ff297eb32868d0010c6be.gif)")
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
    		show_top_players(player_df, keeper_df ,league,position,season)


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
	col1,col2,col3,col4 = st.beta_columns(4)
	with col2:

		fig = make_subplots(rows=2, cols=2,subplot_titles=("Goals scored", "Assists", "Goals per 90", "Assists per 90"), horizontal_spacing=0.30)

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
				x = season_data[[attribute]].values,
				y = season_data.squad,
				orientation ='h'
		))
	fig.update_layout(yaxis={'categoryorder':'total ascending'})
	fig.update_layout(height=500, width=600)
	fig.update_layout(
    title={
        'text': 'Teams' + ' ' + str(attribute) + ' ' + 'stats of the' + ' ' + str(season) + ' ' + ' season',
        'y':0.98,
        'x':0.40,
        'xanchor': 'center',
        'yanchor': 'top'})
	st.plotly_chart(fig)


def visualize_players(df, keeper_df, player_1, player_2, season):
	season_data = df[df['Season'] == season]
	player_1_data = season_data[season_data['player'] == player_1]
	player_2_data = season_data[season_data['player'] == player_2] 
	season_data_gk = keeper_df[keeper_df['Season'] == season]
	player_1_data_gk = season_data_gk[season_data_gk['player'] == player_1]
	player_2_data_gk = season_data_gk[season_data_gk['player'] == player_2]
	horizontal_df = pd.concat([player_1_data.T, player_2_data.T],axis =1)
	horizontal_df_gk = pd.concat([player_1_data_gk.T, player_2_data_gk.T],axis =1)
	nn_data_1 = player_1_data.drop(['player','nationality','squad','age','Season','League'],axis=1)
	nn_data_2 = player_2_data.drop(['player','nationality','squad','age','Season','League'],axis=1)
	nn_data_1_gk = player_1_data_gk.drop(['player','nationality','position','squad','age','Season','League'],axis=1)
	nn_data_2_gk = player_2_data_gk.drop(['player','nationality','position','squad','age','Season','League'],axis=1)
	label_encoder = preprocessing.LabelEncoder()
	nn_data_1['position'] = label_encoder.fit_transform(nn_data_1['position'])
	nn_data_2['position'] = label_encoder.fit_transform(nn_data_2['position'])
	col1,col2,col3 = st.beta_columns(3)
	col4,col5,col6 = st.beta_columns(3)
	with col1:
		if ((player_1_data['position'].values == 'GK') and (player_2_data['position'].values == 'GK')):
			st.write('Goalkeeper detailed stats')
			st.dataframe(horizontal_df_gk)
		else:
			st.write('Players detailed stats')
			st.dataframe(horizontal_df)
	with col3:
		if ((player_1_data['position'].values == 'GK') and (player_2_data['position'].values == 'GK')):
			st.write('Similarity percentage between two players')
			value = cosine_similarity(nn_data_1_gk,nn_data_2_gk) * 100
			st.write(value)
		else:
			st.write('Similarity percentage between two players')
			value = cosine_similarity(nn_data_1,nn_data_2) * 100
			st.write(value)

	player_1_data['position']= player_1_data['position'].astype(str)
	pos = player_1_data['position'].str.lower()
	position_match = pos.str.contains(str(player_2_data['position'].values).lower(),regex=True)
	if position_match.iloc[0] == 1:
		with col4:

			if ((player_1_data['position'].values == 'FW') or (player_1_data['position'].values == 'FW,MF') or (player_1_data['position'].values == 'MF,FW')):
				data = [go.Scatterpolar(
  					r = [player_1_data['goals_assists_per90'].values[0],player_1_data['goals_per_shot_on_target'].values[0],player_1_data['xg_xa_per90'].values[0],player_1_data['sca_per90'].values[0],player_1_data['npxg_xa_per90'].values[0],player_1_data['npxg_net'].values[0],player_1_data["gca_per90"].values[0]],
  					theta = ['goals and assists per90','goals per shot on target','Exp Goals and assists per90','shot creation per 90','Non penalty Exp goals and assists per 90','Non penalty Exp goals','goal creation per 90'],
  					fill = 'toself',
  					name= str(player_1),
     				line =  dict(
                	color = 'orange'
            			)
        			),
    				go.Scatterpolar(
    					r = [player_2_data['goals_assists_per90'].values[0],player_2_data['goals_per_shot_on_target'].values[0],player_2_data['xg_xa_per90'].values[0],player_2_data['sca_per90'].values[0],player_2_data['npxg_xa_per90'].values[0],player_2_data['npxg_net'].values[0],player_2_data["gca_per90"].values[0]],
  						theta = ['goals and assists per90','goals per shot on target','Exp Goals and assists per90','shot creation per 90','Non penalty Exp goals and assists per 90','Non penalty Exp goals','goal creation per 90'],
  						fill = 'toself',
  						name= str(player_2),
     					line =  dict(
                		color = 'red'
            			)
					)]

				layout = go.Layout(
  					polar = dict(
    				radialaxis = dict(
      				visible = True,
      				range = [0, 10]
    					)
  					),
  					showlegend = True,
  					title = "{} vs {} Stats Comparison".format(str(player_1), str(player_2))
  					)
				fig = go.Figure(data=data, layout=layout)
				st.plotly_chart(fig)
				with col6:

					add_attributes = ['shots_on_target_pct','passes_into_penalty_area','passes_into_final_third','progressive_passes','pressures','dribbles_completed_pct','passes_intercepted']
					add_attributes_1 = player_1_data[['shots_on_target_pct','passes_into_penalty_area','passes_into_final_third','progressive_passes','pressures','dribbles_completed_pct','passes_intercepted','fouled']]
					add_attributes_2 = player_2_data[['shots_on_target_pct','passes_into_penalty_area','passes_into_final_third','progressive_passes','pressures','dribbles_completed_pct','passes_intercepted','fouled']]
					horizontal_da = pd.concat([add_attributes_1.T, add_attributes_2.T],axis =1)
					horizontal_da.columns = ['player_1','player_2']
					fig = go.Figure(data=[go.Bar(
						name = player_1,
						x = horizontal_da.index,
						y = horizontal_da['player_1'].values
						),
						go.Bar(
							name = player_2,
							x = horizontal_da.index,
							y = horizontal_da['player_2'].values
							)
						])
					fig.update_layout(yaxis={'categoryorder':'total ascending'})
					fig.update_layout(barmode='stack')
					fig.update_layout(
    					title={
        				'text': 'Additional stats of the' + ' ' + str(player_1) + ' and ' +str(player_2) + ' ' + str(season),
        				'y':0.98,
        				'x':0.40,
        				'xanchor': 'center',
        				'yanchor': 'top'})
					st.plotly_chart(fig)

			elif ((player_1_data['position'].values == 'MF') or (player_1_data['position'].values =='MF,DF')):
				data = [go.Scatterpolar(
  					r = [player_1_data['sca_per90'].values[0],player_1_data['gca_per90'].values[0],player_1_data['xg_xa_per90'].values[0],player_1_data['pressure_regain_pct'].values[0],player_1_data['passes_intercepted'].values[0],player_1_data['through_balls'].values[0],player_1_data["npxg_xa_per90"].values[0]],
  					theta = ['shot creation per 90','goals creation per 90','xg xa per 90','pressure regains percentage','passes intercepted','through balls','non penalty expecting GA per 90'],
  					fill = 'toself',
  					name= str(player_1),
     				line =  dict(
            		color = 'orange'
        				)
					),
					go.Scatterpolar(
  					r = [player_2_data['sca_per90'].values[0],player_2_data['gca_per90'].values[0],player_2_data['xg_xa_per90'].values[0],player_2_data['pressure_regain_pct'].values[0],player_2_data['passes_intercepted'].values[0],player_2_data['through_balls'].values[0],player_2_data["npxg_xa_per90"].values[0]],
  					theta = ['shot creation per 90','goals creation per 90','xg xa per 90','pressure regains percentage','passes intercepted','through balls','non penalty expecting GA per 90'],
  					fill = 'toself',
  					name= str(player_2),
     				line =  dict(
            		color = 'red'
        				)
					)]

				layout = go.Layout(
  					polar = dict(
    				radialaxis = dict(
      				visible = True,
      				range = [0, 50]
    				)
  				),
  					showlegend = True,
  					title = "{} vs {} Stats Comparison".format(str(player_1), str(player_2))
				)
				fig = go.Figure(data=data, layout=layout)
				st.plotly_chart(fig)
				with col6:

					add_attributes_1 = player_1_data[['ball_recoveries','passes_into_penalty_area','passes_into_final_third','progressive_passes','pressures','dribbles_completed_pct','passes_received_pct','fouled']]
					add_attributes_2 = player_2_data[['ball_recoveries','passes_into_penalty_area','passes_into_final_third','progressive_passes','pressures','dribbles_completed_pct','passes_received_pct','fouled']]
					horizontal_da = pd.concat([add_attributes_1.T, add_attributes_2.T],axis =1)
					horizontal_da.columns = ['player_1','player_2']
					fig = go.Figure(data=[go.Bar(
						name = player_1,
						x = horizontal_da.index,
						y = horizontal_da['player_1'].values
						),
						go.Bar(
							name = player_2,
							x = horizontal_da.index,
							y = horizontal_da['player_2'].values
							)
						])
					fig.update_layout(yaxis={'categoryorder':'total ascending'})
					fig.update_layout(barmode='stack')
					fig.update_layout(
    					title={
        				'text': 'Additional stats of the' + ' ' + str(player_1) + ' and ' +str(player_2) + ' ' + str(season),
        				'y':0.98,
        				'x':0.40,
        				'xanchor': 'center',
        				'yanchor': 'top'})
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
            		color = 'red'
        				)
					)]

				layout = go.Layout(
  					polar = dict(
    				radialaxis = dict(
      				visible = True,
      				range = [0, 100]
    					)
  					),
  					showlegend = True,
  					title = "{} vs {} Stats Comparison".format(str(player_1), str(player_2))
					)
				fig = go.Figure(data=data, layout=layout)
				st.plotly_chart(fig)
				with col6:

					add_attributes_1 = player_1_data[['ball_recoveries','fouls','dribble_tackles_pct','tackles_def_3rd','pressure_regain_pct','clearances','passes_received_pct']]
					add_attributes_2 = player_2_data[['ball_recoveries','fouls','dribble_tackles_pct','tackles_def_3rd','pressure_regain_pct','clearances','passes_received_pct']]
					horizontal_da = pd.concat([add_attributes_1.T, add_attributes_2.T],axis =1)
					horizontal_da.columns = ['player_1','player_2']
					fig = go.Figure(data=[go.Bar(
						name = player_1,
						x = horizontal_da.index,
						y = horizontal_da['player_1'].values
						),
						go.Bar(
							name = player_2,
							x = horizontal_da.index,
							y = horizontal_da['player_2'].values
							)
						])
					fig.update_layout(yaxis={'categoryorder':'total ascending'})
					fig.update_layout(barmode='stack')
					fig.update_layout(
    					title={
        				'text': 'Additional stats of the' + ' ' + str(player_1) + ' and ' +str(player_2) + ' ' + str(season),
        				'y':0.98,
        				'x':0.40,
        				'xanchor': 'center',
        				'yanchor': 'top'})
					st.plotly_chart(fig)

			elif ((player_1_data['position'].values == 'GK') and (player_2_data['position'].values == 'GK')):
				data = [go.Scatterpolar(
  					r = [player_1_data_gk['goals_against_per90_gk'].values[0],player_1_data_gk['save_pct'].values[0],player_1_data_gk['psnpxg_per_shot_on_target_against'].values[0],player_1_data_gk['pens_saved'].values[0],player_1_data_gk['crosses_stopped_pct_gk'].values[0],player_1_data_gk['psxg_net_gk'].values[0],player_1_data_gk["pens_allowed"].values[0]],
  					theta = ['Goals allowed per 90','Saves percentage','post shot xg saves on target','pens_saved','cross stopped pct','post shot expecting goals','penalty allowed'],
  					fill = 'toself',
  					name= str(player_1),
     				line =  dict(
            		color = 'orange'
        				)
					),
					go.Scatterpolar(
  					r = [player_2_data_gk['goals_against_per90_gk'].values[0],player_2_data_gk['save_pct'].values[0],player_2_data_gk['psnpxg_per_shot_on_target_against'].values[0],player_2_data_gk['pens_saved'].values[0],player_2_data_gk['crosses_stopped_pct_gk'].values[0],player_2_data_gk['psxg_net_gk'].values[0],player_2_data_gk["pens_allowed"].values[0]],
  					theta = ['Goals allowed per 90','Saves percentage','post shot non penalty xg on target','pens_saved','cross stopped pct','post shot expecting goals','penalty allowed'],
  					fill = 'toself',
  					name= str(player_2),
     				line =  dict(
            		color = 'red'
        				)
					)]

				layout = go.Layout(
  					polar = dict(
    				radialaxis = dict(
      				visible = True,
      				range = [0, 10]
    					)
  					),
  					showlegend = True,
  					title = "{} vs {} Stats Comparison".format(str(player_1), str(player_2))
					)
				fig = go.Figure(data=data, layout=layout)
				st.plotly_chart(fig)
				with col6:

					add_attributes_1 = player_1_data_gk[['psxg_gk','saves','clean_sheets','free_kick_goals_against_gk','corner_kick_goals_against_gk','passes_length_avg_gk','avg_distance_def_actions_gk']]
					add_attributes_2 = player_2_data_gk[['psxg_gk','saves','clean_sheets','free_kick_goals_against_gk','corner_kick_goals_against_gk','passes_length_avg_gk','avg_distance_def_actions_gk']]
					horizontal_da = pd.concat([add_attributes_1.T, add_attributes_2.T],axis =1)
					horizontal_da.columns = ['player_1','player_2']
					fig = go.Figure(data=[go.Bar(
						name = player_1,
						x = horizontal_da.index,
						y = horizontal_da['player_1'].values
						),
						go.Bar(
							name = player_2,
							x = horizontal_da.index,
							y = horizontal_da['player_2'].values
							)
						])
					fig.update_layout(yaxis={'categoryorder':'total ascending'})
					fig.update_layout(barmode='stack')
					fig.update_layout(
    					title={
        				'text': 'Additional stats of the' + ' ' + str(player_1) + ' and ' +str(player_2) + ' ' + str(season),
        				'y':0.98,
        				'x':0.40,
        				'xanchor': 'center',
        				'yanchor': 'top'})
					st.plotly_chart(fig)


	else:
		st.header('Please select players who plays in the same position')



def visualize_single_player(df, keeper_df, player_1, season):
	season_data = df[df['Season'] == season]
	player_1_data = season_data[season_data['player'] == player_1]
	season_data_gk = keeper_df[keeper_df['Season'] == season]
	player_1_data_gk = season_data_gk[season_data_gk['player'] == player_1]
	if (player_1_data['position'].values == 'FW'):
		season_data_position = season_data[season_data['position'] == 'FW']
		season_data_position.reset_index(inplace=True,drop=True)

	elif (player_1_data['position'].values == 'FW,MF'):
		season_data_position = season_data[season_data['position'] == 'FW,MF']
		season_data_position.reset_index(inplace=True,drop=True)
	elif (player_1_data['position'].values == 'MF,FW'):
		season_data_position = season_data[season_data['position'] == 'MF,FW']
		season_data_position.reset_index(inplace=True,drop=True)
	elif (player_1_data['position'].values == 'MF'):
		season_data_position = season_data[season_data['position'] == 'MF']
		season_data_position.reset_index(inplace=True,drop=True)
	elif (player_1_data['position'].values == 'MF,DF'):
		season_data_position = season_data[season_data['position'] == 'MF,DF']
		season_data_position.reset_index(inplace=True,drop=True)
	elif (player_1_data['position'].values == 'DF'):
		season_data_position = season_data[season_data['position'] == 'DF']
		season_data_position.reset_index(inplace=True,drop=True)
	elif (player_1_data['position'].values == 'DF,MF'):
		season_data_position = season_data[season_data['position'] == 'DF,MF']
		season_data_position.reset_index(inplace=True,drop=True)
	elif (player_1_data['position'].values == 'GK'):
		season_data_position = season_data_gk[season_data_gk['position'] == 'GK']
		season_data_position.reset_index(inplace=True,drop=True)


	col1,col2,col3 = st.beta_columns(3)
	col4,col5,col6 = st.beta_columns(3)
	nn_data = season_data_position.drop(['player','position','nationality','squad','age'],axis=1)
	label_encoder = preprocessing.LabelEncoder()
	nn_data['Season'] = label_encoder.fit_transform(nn_data['Season'])
	nn_data['League'] = label_encoder.fit_transform(nn_data['League'])
	model = NearestNeighbors(algorithm = 'ball_tree', n_neighbors=12)
	model.fit(nn_data)
	ind = season_data_position[season_data_position['player']==player_1].index.tolist()[0]
	indices = model.kneighbors(nn_data)[1]
	df = pd.DataFrame(columns=['Recommended players'])
	def recommend_me(ind,df):
		for i in indices[ind][1:]:
			df = df.append({'Recommended players': (season_data_position.iloc[i]['player'])},ignore_index=True)
		return df

	with col1:
		if player_1_data['position'].values == 'GK':
			st.write('Goalkeeper detailed stats')
			st.write(player_1_data_gk.T)
		else:
			st.write('Players detailed stats')
			st.write(player_1_data.T)
	
	st.write('\n')

	with col3:
		st.write('Recommended players list of the season')
		st.write(recommend_me(ind,df))
	
	with col4:

		if ((player_1_data['position'].values == 'FW') or (player_1_data['position'].values == 'FW,MF') or (player_1_data['position'].values == 'MF,FW')):

			data = [go.Scatterpolar(
  				r = [player_1_data['goals_assists_per90'].values[0],player_1_data['gca_per90'].values[0],player_1_data['xg_xa_per90'].values[0],player_1_data['sca_per90'].values[0],player_1_data['npxg_xa_per90'].values[0],player_1_data['npxg_net'].values[0],player_1_data["goals_per_shot_on_target"].values[0]],
  				theta = ['shot on target pct','goals creation per 90','xg xa per 90','shot creation per 90','non penalty Exp goals and assists ','non penalty Exp goals','goals per shot on target'],
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
      			range = [0, 10]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution".format(player_1)
			)
			fig = go.Figure(data=data, layout=layout)
			st.plotly_chart(fig)
			with col6:
				add_attributes_1 = player_1_data[['shots_on_target_pct','passes_into_penalty_area','passes_into_final_third','progressive_passes','pressures','dribbles_completed_pct','passes_intercepted']]
				horizontal_da = add_attributes_1.T
				horizontal_da.columns = ['player_1']
				fig = go.Figure(data=[go.Bar(
					name = player_1,
					x = horizontal_da.index,
					y = horizontal_da['player_1'].values
					)
					])
				fig.update_layout(yaxis={'categoryorder':'total ascending'})
				fig.update_layout(height=500,width=550)
				fig.update_layout(
    				title={
        			'text': 'Additional stats of the' + ' ' + str(player_1) + ' ' + str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
				st.plotly_chart(fig)
		
		elif ((player_1_data['position'].values == 'MF') or (player_1_data['position'].values =='MF,DF')):
			data = [go.Scatterpolar(
  				r = [player_1_data['sca_per90'].values[0],player_1_data['gca_per90'].values[0],player_1_data['xg_xa_per90'].values[0],player_1_data['passes_into_final_third'].values[0],player_1_data['passes_into_penalty_area'].values[0],player_1_data['through_balls'].values[0],player_1_data["npxg_xa_per90"].values[0]],
  				theta = ['shot creation per 90','goals creation per 90','xg xa per 90','passes into final third','passes into penalty area','through balls','non penalty goals per90'],
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
      			range = [0, 50]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution".format(player_1)
			)
			fig = go.Figure(data=data, layout=layout)
			st.plotly_chart(fig)
			with col6:
				add_attributes_1 = player_1_data[['ball_recoveries','passes_into_penalty_area','passes_into_final_third','progressive_passes','pressures','dribbles_completed_pct','passes_received_pct','fouled']]
				horizontal_da = add_attributes_1.T
				horizontal_da.columns = ['player_1']
				fig = go.Figure(data=[go.Bar(
					name = player_1,
					x = horizontal_da.index,
					y = horizontal_da['player_1'].values
					)
					])
				fig.update_layout(yaxis={'categoryorder':'total ascending'})
				fig.update_layout(height=500,width=600)
				fig.update_layout(
    				title={
        			'text': 'Additional stats of the' + ' ' + str(player_1) + ' ' + str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
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
      			range = [0, 100]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution".format(player_1)
			)
			fig = go.Figure(data=data, layout=layout)
			st.plotly_chart(fig)
			with col6:

				add_attributes_1 = player_1_data[['ball_recoveries','fouls','dribble_tackles_pct','tackles_def_3rd','pressure_regain_pct','clearances','passes_received_pct']]
				horizontal_da = add_attributes_1.T
				horizontal_da.columns = ['player_1']
				fig = go.Figure(data=[go.Bar(
					name = player_1,
					x = horizontal_da.index,
					y = horizontal_da['player_1'].values
					)
					])
				fig.update_layout(yaxis={'categoryorder':'total ascending'})
				fig.update_layout(height=500,width=600)
				fig.update_layout(
    				title={
        			'text': 'Additional stats of the' + ' ' + str(player_1) + ' ' + str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
				st.plotly_chart(fig)

		elif (player_1_data['position'].values == 'GK'):
			data = [go.Scatterpolar(
  				r = [player_1_data_gk['goals_against_gk'].values[0],player_1_data_gk['goals_against_per90_gk'].values[0],player_1_data_gk['save_pct'].values[0],player_1_data_gk['clean_sheets_pct'].values[0],player_1_data_gk['pens_allowed'].values[0],player_1_data_gk['pens_saved'].values[0],player_1_data_gk["psnpxg_per_shot_on_target_against"].values[0]],
  				theta = ['Goals against GK','Goals against per 90','save percentage','clean sheets percentage','penalty allowed','penalty saved','non penalty post shot expecting  goals'],
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
      			range = [0, 10]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution".format(player_1)
			)
			fig = go.Figure(data=data, layout=layout)
			st.plotly_chart(fig)
			with col6:

				add_attributes_1 = player_1_data_gk[['psxg_gk','saves','clean_sheets','free_kick_goals_against_gk','corner_kick_goals_against_gk','passes_length_avg_gk','avg_distance_def_actions_gk']]
				horizontal_da = add_attributes_1.T
				horizontal_da.columns = ['player_1']
				fig = go.Figure(data=[go.Bar(
					name = player_1,
					x = horizontal_da.index,
					y = horizontal_da['player_1'].values
					)
					])
				fig.update_layout(yaxis={'categoryorder':'total ascending'})
				fig.update_layout(height=500,width=600)
				fig.update_layout(
    				title={
        			'text': 'Additional stats of the' + ' ' + str(player_1) + ' ' + str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
				st.plotly_chart(fig)


def show_top_players(df,keeper_df,league,position,season):
	season_data = df[df['Season'] == season]
	position_data = season_data[season_data['position'] == position]	
	league_data = position_data[position_data['League'] == league]
	season_data_gk = keeper_df[keeper_df['Season'] == season]	
	league_data_gk = season_data_gk[season_data_gk['League'] == league]
	col1,col2,col3 = st.beta_columns(3)
	with col2:

		if ((position == 'FW') or (position == 'FW,MF') or (position == 'MF,FW')):
			league_data['performance_index'] = league_data['goals_assists_per90'] + league_data['gca_per90'] + league_data['xg_xa_per90'] + league_data['passes_into_final_third'] + league_data['passes_into_penalty_area'] + league_data['through_balls'] + league_data['touches_att_pen_area']
			league_data.sort_values(by=['performance_index'], ascending=False,inplace=True)
			league_data['performance_index'] = (league_data['performance_index'] - league_data['performance_index'].min()) / (league_data['performance_index'].max() - league_data['performance_index'].min()) * 9 + 0.43
			df1 = league_data[['player', 'performance_index','squad']]
			df1.reset_index(drop=True,inplace=True)
			st.write(df1)
		elif ((position == 'MF') or (position == 'MF,DF')):
			league_data['performance_index'] = league_data['sca_per90'] + league_data['gca_per90'] + league_data['xg_xa_per90'] + league_data['passes_into_final_third'] + league_data['passes_into_penalty_area'] + league_data['through_balls'] + league_data['progressive_passes']
			league_data.sort_values(by=['performance_index'], ascending=False,inplace=True)
			league_data['performance_index'] = (league_data['performance_index'] - league_data['performance_index'].min()) / (league_data['performance_index'].max() - league_data['performance_index'].min()) * 9 + 0.46
			df1 = league_data[['player', 'performance_index','squad']]
			df1.reset_index(drop=True,inplace=True)
			st.write(df1)
		elif ((position == 'DF') or (position == 'DF,MF')):
			league_data['performance_index'] = league_data['goals'] + league_data['assists'] + league_data['goals_assists_per90'] + league_data['gca_per90'] + league_data['sca_per90'] - league_data['fouls'] - league_data['dispossessed'] + league_data['ball_recoveries'] - league_data['pens_conceded'] + league_data['clearances'] - league_data['errors'] - league_data['miscontrols'] + league_data['interceptions'] + league_data['tackles_won'] + league_data['blocked_shots'] + league_data['blocked_shots_saves'] + league_data['tackles_def_3rd'] + league_data['aerials_won'] - league_data['aerials_lost']
			league_data.sort_values(by=['performance_index'], ascending=False,inplace=True)
			league_data['performance_index'] = (league_data['performance_index'] - league_data['performance_index'].min()) / (league_data['performance_index'].max() - league_data['performance_index'].min()) * 9 + 0.55
			df1 = league_data[['player', 'performance_index','squad']]
			df1.reset_index(drop=True,inplace=True)
			st.write(df1)
		elif position == 'GK':
			league_data_gk['performance_index'] = league_data_gk['saves'] - league_data_gk['goals_against_per90_gk']  + league_data_gk['save_pct'] + league_data_gk['clean_sheets_pct'] - league_data_gk['pens_allowed'] - league_data_gk['free_kick_goals_against_gk'] - league_data_gk['corner_kick_goals_against_gk'] + league_data_gk['psnpxg_per_shot_on_target_against']/ league_data_gk['saves'] + league_data_gk['crosses_stopped_pct_gk']
			league_data_gk.sort_values(by=['performance_index'], ascending=False,inplace=True)
			league_data_gk['performance_index'] = (league_data_gk['performance_index'] - league_data_gk['performance_index'].min()) / (league_data_gk['performance_index'].max() - league_data_gk['performance_index'].min()) * 9 + 0.68
			league_data_gk['performance_index'].fillna(0,inplace=True)
			df1 = league_data_gk[['player', 'performance_index','squad']]
			df1.reset_index(drop=True,inplace=True)
			st.write(df1)

def visualize_teams_players(df,season,player,play):
	player_data = df[df['player'] == player]
	season_data = player_data[player_data['Season'] == season]
	squad = season_data['squad'].values
	col1,col2,col3 = st.beta_columns(3)
	colors = ['red','orange']*6
	if season_data['position'].values == 'GK':
		st.write("This tool is not for goalkeepers team contribution")
		return None
	if play == 'Attacking':
		columns = ['goals_x','goals_y','assists_x','assists_y','gca_x','gca_y','sca_x','sca_y','passes_into_final_third_x','passes_into_final_third_y','passes_into_penalty_area_x','passes_into_penalty_area_y']
		att = season_data[columns]
		new_col = ['goals by team','goals by player','assists by team','assists by player','goal creation team','goal creation player','shot creation team','shot creation player','final 3 pass team','final 3 pass player','pass into penalty team','pass into penalty player']
		with col2:
			fig = go.Figure(data=[go.Bar(
					name = player,
					x = new_col,
					y = att.iloc[0],
					marker_color = colors
					)
					])
			fig.update_layout(height=500,width=800)
			fig.update_layout(
    				title={
        			'text': 'Comparision stats of the' + ' ' + str(squad) +' '+'and' + ' ' + str(player) + ' ' +' '+ str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
			st.plotly_chart(fig)
	elif play == 'Passing':
		columns = ['passes_completed_x','passes_completed_y','passes_completed_long_x','passes_completed_long_y','dribbles_completed_x','dribbles_completed_y','progressive_passes_x','progressive_passes_y','through_balls_x','through_balls_y','passes_into_penalty_area_x','passes_into_penalty_area_y']
		att = season_data[columns]
		new_col = ['passes completed team','passes completed player','long passes completed team','long passes completed player','dribbles completed team','dribbles completed player','progressive passes team','progressive passes player','through balls team','through balls player','pass into penalty team','pass into penalty player']
		with col2:
			fig = go.Figure(data=[go.Bar(
					name = player,
					x = new_col,
					y = att.iloc[0],
					marker_color = colors
					)
					])
			fig.update_layout(height=500,width=800)
			fig.update_layout(
    				title={
        			'text': 'Comparision stats of the' + ' ' + str(squad) +' '+'and' + ' ' + str(player) + ' ' +' '+ str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
			st.plotly_chart(fig)
	elif play == 'Defending':
		columns = ['blocked_passes_x','blocked_passes_y','blocked_shots_x','blocked_shots_y','blocks_x','blocks_y','clearances_x','clearances_y','tackles_def_3rd_x','tackles_def_3rd_y','pressure_regains_x','pressure_regains_y']
		att = season_data[columns]
		new_col = ['blocked passes team','blocked passes player','blocked shots team','blocked shots player','blocks by team','blocks by player','clearances by team','clearances by player','tackles in def 3 team','tackles in def 3 player','pressure regains team','pressure regains player']
		with col2:
			fig = go.Figure(data=[go.Bar(
					name = player,
					x = new_col,
					y = att.iloc[0],
					marker_color = colors
					)
					])
			fig.update_layout(height=500,width=800)
			fig.update_layout(
    				title={
        			'text': 'Comparision stats of the' + ' ' + str(squad) +' '+'and' + ' ' + str(player) + ' ' +' '+ str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
			st.plotly_chart(fig)

def visualize_attribute_players(player_df, keeper_df, league, season, attribute):
	if attribute in keeper_df.columns:
		league_data = keeper_df[keeper_df['League'] == league]
		X = 'minutes_gk'
	else:
		league_data = player_df[player_df['League'] == league]
		X = 'minutes'
	season_data = league_data[league_data['Season'] == season]
	df = season_data.sort_values(by=[attribute])
	fig = px.scatter(df, x=X ,y=attribute, color = "squad", size = attribute , hover_data=['player'])
	fig.update_layout(height=500,width=800)
	fig.update_layout(
    				title={
        			'text': 'Top players of the' + ' ' + str(league) +' '+'having' +' '+ str(attribute) +' ' + 'in' +' '+ str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
	st.plotly_chart(fig)



if __name__ == "__main__":
    main()





