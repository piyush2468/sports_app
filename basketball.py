import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy import stats
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

hide_streamlit_style = """
					<style>
					#MainMenu {visibility: hidden;}
					footer {visibility: hidden;}
					</style>
					"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

Games_url = ('s3://nbabdata/games.csv')
Games_details_url = ('s3://nbabdata/games_details.csv')
players_url = ('s3://nbabdata/players.csv')
rankings_url = ('s3://nbabdata/ranking.csv')
teams_url = ('s3://nbabdata/teams.csv')

@st.cache(allow_output_mutation=True)
def load_data4():
    df = pd.read_csv(Games_url)
    return df

@st.cache(allow_output_mutation=True)
def load_data5():
    df = pd.read_csv(Games_details_url)
    return df

@st.cache(allow_output_mutation=True)
def load_data6():
    df = pd.read_csv(players_url)
    return df

@st.cache(allow_output_mutation=True)
def load_data7():
    df = pd.read_csv(rankings_url)
    return df

@st.cache(allow_output_mutation=True)
def load_data8():
    df = pd.read_csv(teams_url)
    return df

games_df = load_data4()
games_details_df = load_data5()
players_df = load_data6()
rankings_df = load_data7()
teams_df = load_data8()

def main():

	
	analysis = ['Match analysis','Match Details analysis','Top attributes players']
	play_type = ['Attacking', 'Defending']
	options = ['Single Player Analysis','Players Comparision']
	teams = rankings_df['TEAM'].unique()
	unique = games_details_df['PLAYER_NAME'].unique()
	equiv = {12020:202021, 12019:201920, 22019:201920, 12018:201819, 22018:201819, 12017:201718, 22017:201718, 12016:201617, 22016:201617, 12015:201516, 22015:201516, 12014:201415, 22014:201415, 12013:201314, 22013:201314, 12012:201213, 22012:201213, 12011:201112, 22011:201112, 12010:201011, 22010:201011, 12009:200910, 22009:200910, 12008:200809, 22008:200809, 12007:200708, 22007:200708, 12006:200607, 22006:200607, 12005:200506, 22005:200506, 12004:200405, 22004:200405, 12003:200304, 22003:200304, 12002:200203, 22002:200203}
	rankings_df["Season"] = rankings_df["SEASON_ID"].map(equiv).astype(int)
	seasons = rankings_df['Season'].unique()
	teams_eq = {1610612737:'Hawks',1610612738:'Celtics',1610612740:'Pelicans',1610612741:'Bulls',1610612742:'Mavericks',1610612743:'Nuggets',1610612745:'Rockets',1610612746:'Clippers',1610612747:'Lakers',1610612748:'Heat',1610612749:'Bucks',1610612750:'Timberwolves',1610612751:'Nets',1610612752:'Knicks',1610612753:'Magic',1610612754:'Pacers',1610612755:'76ers',1610612756:'Suns',1610612757:'Trail Blazers',1610612758:'Kings',1610612759:'Spurs',1610612760:'Thunder',1610612761:'Raptors',1610612762:'Jazz',1610612763:'Grizzlies',1610612764:'Wizards',1610612765:'Pistons',1610612766:'Hornets',1610612739:'Cavaliers',1610612744:'Warriors'}
	games_df["HomeTeam"] = games_df["HOME_TEAM_ID"].map(teams_eq).astype(str)
	games_df["VisitingTeam"] = games_df["VISITOR_TEAM_ID"].map(teams_eq).astype(str)
	games_df['Match'] = games_df['HomeTeam'] + ' '+'vs' + ' '+ games_df['VisitingTeam']
	stats_cols = {
    'FGM':'Field Goals Made',
    'FGA':'Field Goals Attempted',
    'FG_PCT':'Field Goal Percentage',
    'FG3M':'Three Pointers Made',
    'FG3A':'Three Pointers Attempted',
    'FG3_PCT':'Three Point Percentage',
    'FTM':'Free Throws Made',
    'FTA':'Free Throws Attempted',
    'FT_PCT':'Free Throw Percentage',
    'OREB':'Offensive Rebounds',
    'DREB':'Defensive Rebounds',
    'REB':'Rebounds',
    'AST':'Assists',
    'TO':'Turnovers',
    'STL':'Steals',
    'BLK':'Blocked Shots',
    'PF':'Personal Foul',
    'PTS':'Points',
    'PLUS_MINUS':'Plus-Minus'
	}
	stat_col = {
	'PTS_home': 'Points made HomeTeam',
	'PTS_away': 'Points made AwayTeam',
	'FG_PCT_home':'Field Goal Percentage HomeTeam',
	'FT_PCT_home':'Free Throw Percentage HomeTeam',
	'FG3_PCT_home': 'Three Point Percentage HomeTeam',
	'AST_home': 'Assists HomeTeam',
	'REB_home': 'Rebounds HomeTeam',
	'FG_PCT_away':'Field Goal Percentage AwayTeam',
	'FT_PCT_away':'Free Throw Percentage AwayTeam',
	'FG3_PCT_away': 'Three Point Percentage AwayTeam',
	'AST_away': 'Assists AwayTeam',
	'REB_away': 'Rebounds AwayTeam'
	}
	attributes = ['Field Goal Percentage HomeTeam','Free Throw Percentage HomeTeam','Three Point Percentage HomeTeam','Assists HomeTeam','Rebounds HomeTeam','Field Goal Percentage AwayTeam','Free Throw Percentage AwayTeam','Three Point Percentage AwayTeam','Assists AwayTeam','Rebounds AwayTeam']
	attributes_players = ['Three Pointers Made','Free Throws Made','Offensive Rebounds','Defensive Rebounds','Rebounds','Assists','Turnovers','Steals','Blocked Shots','Personal Foul']
	games_details_df_new = rename_df(games_details_df, stats_cols) 
	games_df_new = rename_df(games_df, stat_col) 
	page = st.sidebar.selectbox("Choose a page", ["Homepage", "Stats Exploration", "Players Comparision", "Scout Watch"])
	if page == "Homepage":
		st.header("This is an advance basketball data explorer.")
		st.write("Please select a page on the left for different analysis.")
		st.markdown("![Alt Text](https://media.tenor.com/images/07002ec173d710cf8309cc1a91e0f7d9/tenor.gif)")
	elif page == "Stats Exploration":
		st.title("Advanced Match Stats Exploration")
		option = st.selectbox('Choose the type of analysis', analysis )
		if option == 'Match analysis':
			col1,col2,col3= st.beta_columns(3)
			col4,col5,col6 = st.beta_columns(3)
			with col1:
				match= st.selectbox("Choose a Match", games_df_new['Match'].unique())
			with col3:
				season = st.selectbox("Choose a season", games_df_new['SEASON'].unique())
			with col5:
				submit = st.button('Submit')
			if submit:
				visualize_matches(games_df_new,season,match)
		elif option == 'Match Details analysis':
			st.header('Choose the player for his stats in a match')
			col1,col2,col3= st.beta_columns(3)
			col4,col5,col6 = st.beta_columns(3)
			with col1:
				player = st.selectbox("Type in to choose a player", unique)
			with col2:
				match = st.selectbox('Type in to choose the match', games_df_new['Match'].unique())
			with col3:
				season = st.selectbox("Choose the season", games_df_new['SEASON'].unique())
			with col5:
				submit = st.button('Submit')
			if submit:
				visualize_players(games_details_df_new, games_df_new, season, player, match)
		elif option == 'Top attributes players':
			st.header('Choose the attribute for top players of the season')
			col1,col2,col3= st.beta_columns(3)
			col4,col5,col6 = st.beta_columns(3)
			with col1:
				att = st.selectbox("Type in to choose the attribute", attributes_players)
			with col3:
				season = st.selectbox("Choose the season", games_df_new['SEASON'].unique())
			with col5:
				submit = st.button('Submit')
			if submit:
				visualize_top_players(games_details_df_new, games_df_new, season, att)

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
				player_season = st.selectbox("Choose the season", games_df_new['SEASON'].unique())
			visualize_two_players(games_details_df_new, games_df_new, player_1, player_2, player_season)
		elif sub == 'Single Player Analysis':
			with col10:
				player_1 = st.selectbox("Choose the player for analysis", unique)
			with col11:
				player_season = st.selectbox("Choose the season", games_df_new['SEASON'].unique())
			visualize_single_player(games_details_df_new,games_df_new, player_1, player_season)	
        	
	elif page == "Scout Watch":
		st.title('Top players of NBA and their ratings of the season')
		season = st.selectbox("Choose the season",games_df_new['SEASON'].unique())
		submit = st.button('Submit')
		if submit:
			show_top_players(games_details_df_new,games_df_new,season)





def visualize_matches(df,season,match):
	season_df = df[df['SEASON'] == season]
	match_df = season_df[season_df['Match'] == match]
	match_df = match_df.drop(['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','TEAM_ID_home','SEASON','TEAM_ID_away'],axis=1)
	st.subheader('Detailed Stats')
	st.write(match_df)
	attributes = ['Points made HomeTeam','Points made AwayTeam','Assists HomeTeam','Rebounds HomeTeam','Assists AwayTeam','Rebounds AwayTeam']
	col1,col2,col3 = st.beta_columns(3)
	rel_df = match_df[attributes]
	sums = rel_df.sum()
	colors = ['red','orange']*5
	with col1:
		fig = go.Figure(data=[go.Bar(
					name = match,
					x = attributes,
					y = sums,
					marker_color = colors
					)
					])
		fig.update_layout(height=500,width=800)
		fig.update_layout(
    				title={
        			'text': 'Summary stats of all the' + ' ' + ' ' + str(match) + ' ' +'played in '+' '+ str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
		st.plotly_chart(fig)
		

def visualize_players(games_details_df_new, games_df_new, season, player, match):
	player_df = games_details_df_new[games_details_df_new['PLAYER_NAME'] == player]
	merged_df = player_df.merge(games_df_new, on='GAME_ID')
	merged_match_df = merged_df[merged_df['Match'] == match]
	merge_df = merged_match_df[merged_match_df['SEASON'] == season]
	st.subheader('Detailed Stats')
	st.write(merge_df.drop(['GAME_ID','TEAM_ID','PLAYER_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','TEAM_ID_home','TEAM_ID_away'],axis=1))
	attributes = ['Points made HomeTeam','Points made AwayTeam','Assists HomeTeam','Rebounds HomeTeam','Assists AwayTeam','Rebounds AwayTeam','Field Goals Made','Field Goals Attempted','Three Pointers Made','Three Pointers Attempted','Free Throws Made','Free Throws Attempted','Offensive Rebounds','Defensive Rebounds','Rebounds','Assists','Turnovers','Steals','Blocked Shots','Personal Foul','Points',]
	col1,col2,col3 = st.beta_columns(3)
	rel_df = merge_df[attributes]
	sums = rel_df.sum()
	colors = ['red','orange']*10
	with col1:
		fig = go.Figure(data=[go.Bar(
					name = match,
					x = attributes,
					y = sums,
					marker_color = colors
					)
					])
		fig.update_layout(height=500,width=800)
		fig.update_layout(
    				title={
        			'text': 'Summary of all the' + ' '+ str(player) + ' '+'stats' +' '+ str(match) + ' ' +'played in '+' '+ str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
		st.plotly_chart(fig)
	ppg = rel_df['Points'].sum() / len(rel_df)
	ppg_ht = rel_df['Points made HomeTeam'].sum() / len(rel_df)
	ppg_at = rel_df['Points made AwayTeam'].sum() / len(rel_df)
	apg = rel_df['Assists'].sum() / len(rel_df)
	apg_ht = rel_df['Assists HomeTeam'].sum() / len(rel_df)
	apg_at = rel_df['Assists AwayTeam'].sum() / len(rel_df)
	col4,col5,col6,col7,col8,col9 =st.beta_columns(6)
	with col4:
		st.write('Points per game by player {}'.format(str(ppg)))
	with col5:
		st.write('Points per game by HT {}'.format(str(ppg_ht)))
	with col6:
		st.write('Points per game by AT {}'.format(str(ppg_at)))
	with col7:
		st.write('Assists per game by player {}'.format(str(apg)))
	with col8:
		st.write('Assists per game by HT {}'.format(str(apg_ht)))
	with col9:
		st.write('Assists per game by AT {}'.format(str(apg_at)))

def rename_df(df, col_dict):
    cols = df.columns
    new_cols = [(col_dict[c] if c in col_dict else c) for c in cols]
    df.columns = new_cols
    return df

def visualize_two_players(games_details_df_new, games_df_new, player_1, player_2, player_season):
	player1_df = games_details_df_new[games_details_df_new['PLAYER_NAME'] == player_1]
	merged1_df = player1_df.merge(games_df_new, on='GAME_ID')
	merge1_df = merged1_df[merged1_df['SEASON'] == player_season]
	player2_df = games_details_df_new[games_details_df_new['PLAYER_NAME'] == player_2]
	merged2_df = player2_df.merge(games_df_new, on='GAME_ID')
	merge2_df = merged2_df[merged2_df['SEASON'] == player_season]
	attributes = ['Three Pointers Made','Free Throws Made','Offensive Rebounds','Defensive Rebounds','Rebounds','Assists','Turnovers','Steals','Blocked Shots','Personal Foul']
	col1,col2,col3,col4 = st.beta_columns(4)
	rel1_df = merge1_df[attributes]
	sums1 = rel1_df.sum()
	rel2_df = merge2_df[attributes]
	sums2 = rel2_df.sum()

	with col1:
		data = [go.Scatterpolar(
  					r = sums1,
  					theta = attributes,
  					fill = 'toself',
  					name= str(player_1),
     				line =  dict(
            		color = 'orange'
        				)
					),
					go.Scatterpolar(
  					r = sums2,
  					theta = attributes,
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
      				range = [0, 1000]
    				)
  				),
  					showlegend = True,
  					title = "{} vs {} Summary Stats Comparison of {}".format(str(player_1), str(player_2), player_season)
				)
		fig = go.Figure(data=data, layout=layout)
		st.plotly_chart(fig)
		merge1_df = merge1_df.drop(['GAME_ID','TEAM_ID','PLAYER_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','TEAM_ID_home','TEAM_ID_away'],axis=1)
		merge2_df = merge2_df.drop(['GAME_ID','TEAM_ID','PLAYER_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','TEAM_ID_home','TEAM_ID_away'],axis=1)
		horizontal_df = pd.concat([merge1_df.T, merge2_df.T],axis =1)
	with col4:
		colors = ['red','orange']*10
		attributes = ['Field Goals Made','Field Goals Attempted','Three Pointers Made','Three Pointers Attempted','Free Throws Made','Free Throws Attempted','Points']
		rel1_df = merge1_df[attributes]
		sums1 = rel1_df.sum()
		rel2_df = merge2_df[attributes]
		sums2 = rel2_df.sum()
		fig = go.Figure(data=[go.Bar(
					name = player_1,
					x = attributes,
					y = sums1
					),
					go.Bar(
							name = player_2,
							x = attributes,
							y = sums2
							)
					])
		fig.update_layout(yaxis={'categoryorder':'total ascending'})
		fig.update_layout(barmode='stack')
		fig.update_layout(height=500,width=550)
		fig.update_layout(
    				title={
        			'text': 'Additional summary stats of the' + ' ' + str(player_1) + ' '+'and'+' '+str(player_2)+' '+'in ' +' '+ str(player_season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
		st.plotly_chart(fig)
	st.subheader('Match by match details')
	st.write(horizontal_df)

def visualize_single_player(games_details_df_new,games_df_new, player_1, player_season):
	player_df = games_details_df_new[games_details_df_new['PLAYER_NAME'] == player_1]
	merged_df = player_df.merge(games_df_new, on='GAME_ID')
	merge_df = merged_df[merged_df['SEASON'] == player_season]
	attributes = ['Three Pointers Made','Free Throws Made','Offensive Rebounds','Defensive Rebounds','Rebounds','Assists','Turnovers','Steals','Blocked Shots','Personal Foul']
	attributes_name = ['PLAYER_NAME','Three Pointers Made','Free Throws Made','Offensive Rebounds','Defensive Rebounds','Rebounds','Assists','Turnovers','Steals','Blocked Shots','Personal Foul']
	col1,col2,col3,col4 = st.beta_columns(4)
	col5,col6 = st.beta_columns(2)
	all_merged_df = games_details_df_new.merge(games_df_new, on='GAME_ID')
	all_merge_df = all_merged_df[all_merged_df['SEASON'] == player_season]
	all_rel_df = all_merge_df[attributes_name]
	group_df = all_rel_df.groupby('PLAYER_NAME').sum().reset_index()
	nn_data = group_df.drop('PLAYER_NAME',axis=1)
	model = NearestNeighbors(algorithm = 'ball_tree', n_neighbors=12)
	model.fit(nn_data)
	ind = group_df[group_df['PLAYER_NAME']==player_1].index.tolist()[0]
	indices = model.kneighbors(nn_data)[1]
	df = pd.DataFrame(columns=['Recommended players'])
	def recommend_me(ind,df):
		for i in indices[ind][1:]:
			df = df.append({'Recommended players': (group_df.iloc[i]['PLAYER_NAME'])},ignore_index=True)
		return df
	with col6:
		st.write('Recommended players list of the season')
		st.write(recommend_me(ind,df))

	rel_df = merge_df[attributes]
	sums = rel_df.sum()
	with col1:
		data = [go.Scatterpolar(
  				r = sums,
  				theta = attributes,
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
      			range = [0, 1000]
    			)
  			),
  				showlegend = True,
  				title = "{} stats distribution in {}".format(player_1,player_season)
			)
		fig = go.Figure(data=data, layout=layout)
		st.plotly_chart(fig)
	with col4:
		colors = ['red','orange']*10
		attributes = ['Field Goals Made','Field Goals Attempted','Three Pointers Made','Three Pointers Attempted','Free Throws Made','Free Throws Attempted','Points']
		rel_df = merge_df[attributes]
		sums = rel_df.sum()
		fig = go.Figure(data=[go.Bar(
					name = player_1,
					x = attributes,
					marker_color = colors,
					y = sums
					)
					])
		fig.update_layout(yaxis={'categoryorder':'total ascending'})
		fig.update_layout(height=500,width=550)
		fig.update_layout(
    				title={
        			'text': 'Additional summary stats of the' + ' ' + str(player_1) + ' '+'in ' +' '+ str(player_season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
		st.plotly_chart(fig)
	st.subheader('Detailed match by match stats')
	st.write(merge_df)

def show_top_players(games_details_df_new,games_df_new,season):
	merged_df = games_details_df_new.merge(games_df_new, on='GAME_ID')
	season_data = merged_df[merged_df['SEASON'] == season]
	group_df = season_data.groupby('PLAYER_NAME').sum().reset_index()
	group_df['performance_index'] = group_df['Points'] + group_df['Field Goal Percentage'] + group_df['Three Point Percentage'] + group_df['Free Throw Percentage'] + group_df['Offensive Rebounds'] + group_df['Defensive Rebounds'] + group_df['Rebounds'] + group_df['Steals'] + group_df['Assists'] + group_df['Turnovers'] - group_df['Personal Foul']
	group_df.sort_values(by=['performance_index'], ascending=False,inplace=True)
	group_df['performance_index'] = (group_df['performance_index'] - group_df['performance_index'].min()) / (group_df['performance_index'].max() - group_df['performance_index'].min()) * 9 + 0.46
	df1 = group_df[['PLAYER_NAME', 'performance_index']]
	df1.reset_index(drop=True,inplace=True)
	st.write(df1)

def visualize_top_players(games_details_df_new, games_df_new, season, att):
	merged_df = games_details_df_new.merge(games_df_new, on='GAME_ID')
	season_data = merged_df[merged_df['SEASON'] == season]
	group_df = season_data.groupby('PLAYER_NAME').sum().reset_index()
	df = group_df.sort_values(by=[att])
	df.fillna(0,inplace = True)
	fig = px.scatter(df, x='Points' ,y=att, color = "Points", size = att , hover_data=['PLAYER_NAME'])
	fig.update_layout(height=500,width=800)
	fig.update_layout(
    				title={
        			'text': 'Top players of the NBA' + ' ' +' '+'having' +' '+ str(att) +' ' + 'in' +' '+ str(season),
        			'y':0.98,
        			'x':0.40,
        			'xanchor': 'center',
        			'yanchor': 'top'})
	st.plotly_chart(fig)

if __name__ == "__main__":
    main()
