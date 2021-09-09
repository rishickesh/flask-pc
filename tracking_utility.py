import json
import pandas as pd
import numpy as np
import warnings
import json
import requests
import xgboost
import re
from math import sqrt
import ast
import time
from sklearn import preprocessing
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output
from mplsoccer import Pitch, VerticalPitch
from mplsoccer.pitch import Pitch
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import silhouette_score
from mplsoccer.statsbomb import read_event, EVENT_SLUG
from matplotlib import cm
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
import math
import socceraction.spadl as spadl
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, roc_auc_score
import socceraction.spadl.statsbomb as statsbomb
warnings.filterwarnings("ignore")
import pandas as pd
pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import tqdm
import os
import warnings
from math import sqrt
from statistics import mean
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import moviepy.video.io.ImageSequenceClip
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import datetime
import socceraction.vaep.formula as vaepformula
import matplotlib.colors as colors
import socceraction.xthreat as xthreat
import csv
from LaurieOnTracking import Metrica_IO as mio
from LaurieOnTracking import Metrica_PitchControl as mpc
from LaurieOnTracking import Metrica_Velocities as mvelo
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pyreadr
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from collections import Counter 
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull,Delaunay
#from LaurieOnTracking import Metrica_Viz as mviz
params=mpc.default_model_params()


image_folder = "/Users/rishickesh/Downloads/"
data_folder = image_folder


def find_forma_2nd(team, ind, goalie, tracking):
    h_xs = tracking[[c for c in tracking.columns if team in c and '_x' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and goalie != c[5:-2]]].iloc[ind].reset_index()
    h_ys = tracking[[c for c in tracking.columns if team in c and '_y' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and goalie != c[5:-2]]].iloc[ind].reset_index()

    #print(h_xs)
    
    period = tracking.loc[ind].period
    #print(period)
    
    h_xs=h_xs.dropna()
    h_ys=h_ys.dropna()

    le = preprocessing.LabelEncoder()

    h_xs.columns = ['index_x','x']
    h_ys.columns = ['index_y','y']
    h_s=pd.concat([h_xs, h_ys], axis=1, sort=False)
    if (brist_attack_side == "left" and period == 1) or (brist_attack_side == "right" and period == 2):
        
        if match_params["0"] == team:
            h_s = h_s.sort_values(by=['x'], ascending = True)
        else:
            h_s = h_s.sort_values(by=['x'], ascending = False)
    else:
        if match_params["0"] == team:
            h_s = h_s.sort_values(by=['x'], ascending = False)
        else:
            h_s = h_s.sort_values(by=['x'], ascending = True)

    arr = h_s['x'].values.reshape(-1,1)

    sil = []
    kmax = 4

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(3, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(arr)
        labels = kmeans.labels_
        sil.append(silhouette_score(arr, labels, metric = 'euclidean'))

    if sil[0]>sil[1]:
        k = 3
    else:
        k = 4

    kmeans = KMeans(n_clusters=k,random_state=0)
    kmeans.fit(arr)
    y_kmeans = kmeans.predict(arr)

    for i in range(0,len(y_kmeans)):
        y_kmeans[i] = str(y_kmeans[i])
    
    h_s['cluster']=y_kmeans
    h_s=h_s[['index_x','x','y','cluster']]

    c_home = Counter(h_s['cluster'].values)

    formation_home = Counter(c_home.elements()).values()
    
    s=""
    
    for i in formation_home:
        s=s+str(i)
        s=s+"-"
    s=s[:-1]
    
    return s, h_s, k


def isInside(circle_x, circle_y, rad, x, y): 
      
    # Compare radius of circle 
    # with distance of its center 
    # from given point 
    if ((x - circle_x) * (x - circle_x) + 
        (y - circle_y) * (y - circle_y) <= rad * rad): 
        return True
    else: 
        return False

def add_circle(x,hometeam,fig,ax):
    ax.plot( hometeam['ball_x'], hometeam['ball_y'], 'ko', MarkerSize=9, alpha=1.0, LineWidth=0)
    r=(hometeam['ball_x']-x)**2+(hometeam['ball_y'])**2
    r=sqrt(r)
    circle1 = plt.Circle((x, 0), r, color='lightgreen')
    ax.add_artist(circle1)
    return r,fig,ax
  
def plot_frame_packing( ind, hometeam, awayteam, figax=None, team_colors=('r','b'), field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=15, PlayerAlpha=0.7, annotate=False, show_img = False ):

    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
    # plot home & away teams in order
    
    jersey_font_size = PlayerMarkerSize*2/3
    ## Get the team in possesion
    if hometeam.ball_owning_team_id == "away":
        te = "away"
    else:
        te = "home"
    
    period = hometeam.period
    
    if te == "away":
        #print(te)
        if period == 1 and brist_attack_side == "left":
            r,fig,ax=add_circle(-80, hometeam, fig,ax)
        else:
            r,fig,ax=add_circle(80, hometeam, fig,ax)
        for team in [awayteam]:
            x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
            y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
            plt.plot( team[x_columns], team[y_columns], 'r'+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
            if include_player_velocities:
                vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color='r', scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
            if annotate:
                [ ax.text( team[x], team[y], x.split('_')[1], fontsize=jersey_font_size, color="w", ha="center", va="center") for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 

        for team in [hometeam]:
            x_column = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] 
            y_column = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y']

        x_columns=[]
        y_columns=[]
        for x,y in zip(x_column,y_column):
            if period == 1 and brist_attack_side == "left":
                circle_start = -80
            else:
                circle_start = 80
            if isInside(circle_start,0,r,team[x],team[y]):
                x_columns.append(x)
                y_columns.append(y)
        plt.plot( team[x_columns], team[y_columns], 'b'+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha )
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
            plt.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color='b', scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
        if annotate:
            [ ax.text( team[x], team[y], x.split('_')[1], fontsize=jersey_font_size, color="w", ha="center", va="center") for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 

    elif te=="home":
        print(te)
        if period == 1 and brist_attack_side == "left":
            r,fig,ax=add_circle(80, hometeam, fig,ax)
        else:
            r,fig,ax=add_circle(-80, hometeam, fig,ax)
            
        for team in [hometeam]:
            x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
            y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
            plt.plot( team[x_columns], team[y_columns], 'b'+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
            if include_player_velocities:
                vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                plt.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color='b', scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
            if annotate:
                [ ax.text( team[x], team[y], x.split('_')[1], fontsize=jersey_font_size, color="w", ha="center", va="center") for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 

        for team in [awayteam]:
            x_column = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] 
            y_column = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y']

        x_columns=[]
        y_columns=[]
        for x,y in zip(x_column,y_column):
            if period == 1 and brist_attack_side == "left":
                circle_start = 80
            else:
                circle_start = -80
            if isInside(circle_start,0,r,team[x],team[y]):
                x_columns.append(x)
                y_columns.append(y)
        plt.plot( team[x_columns], team[y_columns], 'r'+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha )
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
            plt.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color='r', scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
        if annotate:
            [ ax.text( team[x], team[y], x.split('_')[1], fontsize=jersey_font_size, color="w", ha="center", va="center") for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 


    plt.xlim(-54,54)#-54to54
    plt.ylim(-35,35)#-35to35
    plt.savefig("/Users/rishickesh/Downloads/output/Packing/" + str(ind) + ".png", dpi = 150)

    if show_img == False:
        plt.close(fig)
        

def get_Bristol_pressure_events(event_data):
    return event_data[(event_data.team_name != "Bristol City") 
           & ((event_data.type_name == "Dispossessed") |((event_data.type_name == "Ball Receipt*")
        & ((event_data.outcome_name == "Incomplete"))))]



def to_single_playing_direction(match_data):
    '''
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    '''
    
    copy_df = match_data.copy(deep=True)
    
    second_half_idx = copy_df.period.idxmax(2)
    columns = [c for c in copy_df.columns if c[-1].lower() in ['x','y']]
    copy_df.loc[second_half_idx:,columns] *= -1
    return copy_df

#@title
import numpy as np
import scipy.signal as signal

def calc_player_velocities(team, smoothing=True, filter_='moving average', window=7, polyorder=1, maxspeed = 12, team_list = ["Bris", "Birm"]):
    """ calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added
    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)

    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in team_list] )
    #print(player_ids)

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = 0.04
    
    # index of first frame in second half
    second_half_idx = team.period.idxmax(2)
    
    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player+"_x"].diff() / dt
        vy = team[player+"_y"].diff() / dt
        
        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan 
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 

       
        
        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return team

def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    team = team.drop(columns=columns)
    return team



#@title
def plot_pitch( field_dimen = (105.0,68.0), field_color ='green', linewidth=2, markersize=20):
    """ plot_pitch
    
    Plots a soccer pitch. All distance units converted to meters.
    
    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    fig,ax = plt.subplots(figsize=(10,7)) # create a figure 
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color=='green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke' # line color
        pc = 'w' # 'spot' colors
    elif field_color=='white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3,3) # include a border arround of the field of width 3m
    meters_per_yard = 0.9144 # unit conversion from yards to meters
    half_pitch_length = field_dimen[0]/2. # length of half pitch
    half_pitch_width = field_dimen[1]/2. # width of half pitch
    signs = [-1,1] 
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8*meters_per_yard
    box_width = 20*meters_per_yard
    box_length = 6*meters_per_yard
    area_width = 44*meters_per_yard
    area_length = 18*meters_per_yard
    penalty_spot = 12*meters_per_yard
    corner_radius = 1*meters_per_yard
    D_length = 8*meters_per_yard
    D_radius = 10*meters_per_yard
    D_pos = 12*meters_per_yard
    centre_circle_radius = 10*meters_per_yard
    # plot half way line # center circle
    ax.plot([0,0],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
    ax.scatter(0.0,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
    y = np.linspace(-1,1,50)*centre_circle_radius
    x = np.sqrt(centre_circle_radius**2-y**2)
    ax.plot(x,y,lc,linewidth=linewidth)
    ax.plot(-x,y,lc,linewidth=linewidth)
    for s in signs: # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_length,half_pitch_length],[s*half_pitch_width,s*half_pitch_width],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
        # goal posts & line
        ax.plot( [s*half_pitch_length,s*half_pitch_length],[-goal_line_width/2.,goal_line_width/2.],pc+'s',markersize=6*markersize/20.,linewidth=linewidth)
        # 6 yard box
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[box_width/2.,box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[-box_width/2.,-box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*box_length,s*half_pitch_length-s*box_length],[-box_width/2.,box_width/2.],lc,linewidth=linewidth)
        # penalty area
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[area_width/2.,area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[-area_width/2.,-area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*area_length,s*half_pitch_length-s*area_length],[-area_width/2.,area_width/2.],lc,linewidth=linewidth)
        # penalty spot
        ax.scatter(s*half_pitch_length-s*penalty_spot,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
        # corner flags
        y = np.linspace(0,1,50)*corner_radius
        x = np.sqrt(corner_radius**2-y**2)
        ax.plot(s*half_pitch_length-s*x,-half_pitch_width+y,lc,linewidth=linewidth)
        ax.plot(s*half_pitch_length-s*x,half_pitch_width-y,lc,linewidth=linewidth)
        # draw the D
        y = np.linspace(-1,1,50)*D_length # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2-y**2)+D_pos
        ax.plot(s*half_pitch_length-s*x,y,lc,linewidth=linewidth)
        
    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0]/2. + border_dimen[0]
    ymax = field_dimen[1]/2. + border_dimen[1]
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    ax.set_axisbelow(True)
    return fig,ax


def pitch_control_single_location(ind, tracking_attack,tracking_defend,params, target_location, field_dimen = (105.,68.,), poss_team = "Bris", def_team = 'Bren', GK_numbers = ["1","1"]):
    #field_dimen = (105.,68.,) 
    
    ### or set it as 50 while loading the normal EPV
    n_grid_cells_x = 25 
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    
    ### tracking_attack here means the Bris team and tracking_defend is the Opponenet team
    if poss_team == 'Bris':
        attacking_players = initialise_players(tracking_attack.loc[ind], poss_team,params)
        defending_players = initialise_players(tracking_defend.loc[ind], def_team,params)
    else:
        defending_players = initialise_players(tracking_attack.loc[ind],def_team,params)
        attacking_players = initialise_players(tracking_defend.loc[ind],poss_team,params)
    
#     attacking_players = initialise_players(tracking_attack.loc[ind], home_team ,params)
#     defending_players = initialise_players(tracking_defend.loc[ind], away_team,params)

    ball_start_pos = np.array([tracking_attack.loc[ind].ball_x,tracking_attack.loc[ind].ball_y])
    
    attacking_players = check_offsides( attacking_players, defending_players, ball_start_pos, GK_numbers)
    
    PPCFa, PPCFd = calculate_pitch_control_at_target(target_location, attacking_players, defending_players, ball_start_pos, params)
          # check probabilitiy sums within convergence 
    #assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    return PPCFa

#@title
import numpy as np


def initialise_players(team,teamname,params):
    """
    initialise_players(team,teamname,params)
    
    create a list of player objects that holds their positions and velocities from the tracking data dataframe 
    
    Parameters
    -----------
    
    team: row (i.e. instant) of either the home or away team tracking Dataframe
    teamname: team name "Bris" 
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returns
    -----------
    
    team_players: list of player objects for the team at at given instant
    
    """    
    # get player  ids
    player_ids = np.unique( [ c.split('_')[1] for c in team.keys() if c[:4] == teamname ] )
    # create list
    team_players = []
    for p in player_ids:
        # create a player object for player_id 'p'
        team_player__ = player(p,team,teamname,params)
        if team_player__.inframe:
            team_players.append(team_player__)
    return team_players

class player(object):
    """
    player() class
    
    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player
    
    __init__ Parameters
    -----------
    pid: id (jersey number) of player
    team: row of tracking data for team
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    
    methods include:
    -----------
    simple_time_to_intercept(r_final): time take for player to get to target position (r_final) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept
    
    """
    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self,pid,team,teamname,params):
        self.id = pid
        self.teamname = teamname
        self.playername = "%s_%s_" % (teamname,pid)
        self.vmax = params['max_player_speed'] # player max speed in m/s. Could be individualised
        self.reaction_time = params['reaction_time'] # player reaction time in 's'. Could be individualised
        self.tti_sigma = params['tti_sigma'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.get_position(team)
        self.get_velocity(team)
        self.PPCF = 0. # initialise this for later
        
    def get_position(self,team):
        self.position = np.array( [ team[self.playername+'x'], team[self.playername+'y'] ] )
        self.inframe = not np.any( np.isnan(self.position) )
        
    def get_velocity(self,team):
        self.velocity = np.array( [ team[self.playername+'vx'], team[self.playername+'vy'] ] )
        if np.any( np.isnan(self.velocity) ):
            self.velocity = np.array([0.,0.])
    
    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0. # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity*self.reaction_time
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final-r_reaction)/self.vmax
        return self.time_to_intercept

    def probability_intercept_ball(self,T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.tti_sigma * (T-self.time_to_intercept) ) )
        return f
    def improved_time_to_reach_location(self,location):
        Xf=location
        X0=self.position
        V0=self.velocity
        alpha = self.amax/self.vmax
        
        #equations of motion + equation 3 from assumption that the player accelerate 
        #with constant acceleration amax to vmax
        #we have to add abs(t) to make t be positive
        def equations(p):
            vxmax, vymax, t = p
            eq1 = Xf[0] - (X0[0] + vxmax*(abs(t) - (1 - np.exp(-alpha*abs(t)))/alpha)+((1 - np.exp(-alpha*abs(t)))/alpha)*V0[0])
            eq2 = Xf[1] - (X0[1] + vymax*(abs(t) - (1 - np.exp(-alpha*abs(t)))/alpha)+((1 - np.exp(-alpha*abs(t)))/alpha)*V0[1])
            eq3 = np.sqrt(vxmax**2+vymax**2) - self.vmax
            return (eq1,eq2,eq3)
        
        #prediction for three unknowns
        t_predict=np.linalg.norm(Xf-X0)/self.vmax+0.7
        v_predict=self.vmax*(Xf-X0)/np.linalg.norm(Xf-X0)
        vxmax, vymax, t =  fsolve(equations, (v_predict[0], v_predict[1], t_predict))

        self.time_to_reach_location=abs(t)
        
        return(abs(t))
    
    def probability_to_reach_location(self,T):
        f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.ttrl_sigma * (T-self.time_to_reach_location ) ) )
        return f

""" Generate pitch control map """

def default_model_params(time_to_control_veto=3):
    """
    default_model_params()
    
    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
    
    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
    
    
    Returns
    -----------
    
    params: dictionary of parameters required to determine and calculate the model
    
    """
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
    params['max_player_speed'] = 5. # maximum player speed m/s
    params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params['lambda_att'] = 4.3 # ball control parameter for attacking team
    params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params['average_ball_speed'] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = 0.04 # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
    params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    # sigma normal distribution for relevant pitch control
    params['sigma_normal'] = 23.9
    # alpha : dependence of the decision conditional probability by the PPCF
    params['alpha'] = 1.04
    return params

def generate_pitch_control_for_event(event_id, events, tracking_home, tracking_away, params, field_dimen = (106.,68.,), n_grid_cells_x = 50):
    """ generate_pitch_control_for_event
    
    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        
    Returrns
    -----------
        PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)
    """
    # get the details of the event (frame, team in possession, ball_start_position)
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    ball_start_pos = np.array([events.loc[event_id]['Start X'],events.loc[event_id]['Start Y']])
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    xgrid = np.linspace( -field_dimen[0]/2., field_dimen[0]/2., n_grid_cells_x)
    ygrid = np.linspace( -field_dimen[1]/2., field_dimen[1]/2., n_grid_cells_y )
    # initialise pitch control grids for attacking and defending teams 
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if pass_team=='Home':
        attacking_players = initialise_players(tracking_home.loc[pass_frame],'Home',params)
        defending_players = initialise_players(tracking_away.loc[pass_frame],'Away',params)
    elif pass_team=='Away':
        defending_players = initialise_players(tracking_home.loc[pass_frame],'Home',params)
        attacking_players = initialise_players(tracking_away.loc[pass_frame],'Away',params)
    else:
        assert False, "Team in possession must be either home or away"
    # calculate pitch pitch control model at each location on the pitch
    for i in range( len(ygrid) ):
        for j in range( len(xgrid) ):
            target_position = np.array( [xgrid[j], ygrid[i]] )
            PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params)
    # check probabilitiy sums within convergence
    checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
    assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    return PPCFa,xgrid,ygrid

def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params):
    """ calculate_pitch_control_at_target
    
    Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
    
    Parameters
    -----------
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )
    """
    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
        ball_travel_time = 0.0 
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
    
    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
    tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
    
    # check whether we actually need to solve equation 3
    if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0., 1.
    elif tau_min_def-max(ball_travel_time,tau_min_att) >= params['time_to_control_att']:
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1., 0.
    else: 
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
        defending_players = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
        PPCFatt = np.zeros_like( dT_array )
        PPCFdef = np.zeros_like( dT_array )
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * params['lambda_att']
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
                #print("Player: " + str(player.id) + " , from  team: " + player.teamname + " PC contribution: " + str(player.PPCF))
            
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * params['lambda_def']
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFdef[i] += player.PPCF # add to sum over players in the defending team
                #print("Player: " + str(player.id) + " , from  team: " + player.teamname + " PC contribution: " + str(player.PPCF))

            ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
            i += 1
        if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
        return PPCFatt[i-1], PPCFdef[i-1]

#@title
def plot_frame( hometeam, awayteam, figax=None, team_colors=('r','b'), field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False ):
    """ plot_frame( hometeam, awayteam )
    
    Plots a frame of Metrica tracking data (player positions and the ball) on a football pitch. All distances should be in meters.
    
    Parameters
    -----------
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    
    jersey_font_size = PlayerMarkerSize*2/3
    
    if figax is None: # create new pitch 
        fig,ax = plot_pitch(field_color="white", field_dimen=field_dimen)
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
    # plot home & away teams in order
    for team,color in zip( [hometeam,awayteam], team_colors) :
        x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
        y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
        ax.plot( team[x_columns], team[y_columns], color+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
        #print(team[x_columns])
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
            #print(pd.to_numeric(team[x_columns]) , type(team[x_columns]))
            ax.quiver( pd.to_numeric(team[x_columns]), pd.to_numeric(team[y_columns]), pd.to_numeric(team[vx_columns]), pd.to_numeric(team[vy_columns]), color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
        if annotate:
            w=1
            [ ax.text( team[x], team[y], x.split('_')[1], fontsize=jersey_font_size, color="w", ha="center", va="center") for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 
    # plot ball
    ax.plot( hometeam['ball_x'], hometeam['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
    return fig,ax

def plot_events( events, figax=None, field_dimen = (106.0,68), indicators = ['Marker','Arrow'], color='r', marker_style = 'o', alpha = 0.5, annotate=False):
    """ plot_events( events )
    
    Plots Metrica event positions on a football pitch. event data can be a single or several rows of a data frame. All distances should be in meters.
    
    Parameters
    -----------
        events: row (i.e. instant) of the home team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        indicators: List containing choices on how to plot the event. 'Marker' places a marker at the 'Start X/Y' location of the event; 'Arrow' draws an arrow from the start to end locations. Can choose one or both.
        color: color of indicator. Default is 'r' (red)
        marker_style: Marker type used to indicate the event position. Default is 'o' (filled ircle).
        alpha: alpha of event marker. Default is 0.5    
        annotate: Boolean determining whether text annotation from event data 'Type' and 'From' fields is shown on plot. Default is False.
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """

    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax 
    for i,row in events.iterrows():
        if 'Marker' in indicators:
            ax.plot(  row['Start X'], row['Start Y'], color+marker_style, alpha=alpha )
        if 'Arrow' in indicators:
            ax.annotate("", xy=row[['End X','End Y']], xytext=row[['Start X','Start Y']], alpha=alpha, arrowprops=dict(alpha=alpha,arrowstyle="->",color=color),annotation_clip=False)
        if annotate:
            textstring = row['Type'] + ': ' + row['From']
            ax.text( row['Start X'], row['Start Y'], textstring, fontsize=10, color=color)
    return fig,ax
    


def get_dissim(h_xs,a_xs,h_ys,a_ys):
    h_xs=transform(h_xs)
    a_xs=transform(a_xs)
    h_ys=transform(h_ys)
    a_ys=transform(a_ys)

    home_cord=[]
    for i in range(0,len(h_xs)):
        home_cord.append([h_xs[i], h_ys[i]])
    home_cord = np.array(home_cord) 
    away_cord=[]
    for i in range(0,len(a_xs)):
        away_cord.append([a_xs[i], a_ys[i]])
    away_cord = np.array(away_cord) 

    del_home = Delaunay(home_cord)
    del_away = Delaunay(away_cord)

    edges_explicit = np.concatenate((del_home.vertices[:, :2],
                                     del_home.vertices[:, 1:],
                                     del_home.vertices[:, ::2]), axis=0)
    adj = np.zeros((home_cord.shape[0], home_cord.shape[0]))
    adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.

    edges_explicit1 = np.concatenate((del_away.vertices[:, :2],
                                     del_away.vertices[:, 1:],
                                     del_away.vertices[:, ::2]), axis=0)
    adj1 = np.zeros((away_cord.shape[0], away_cord.shape[0]))
    adj1[edges_explicit1[:, 0], edges_explicit1[:, 1]] = 1.
    return np.clip(adj + adj.T, 0, 1) ,np.clip(adj1 + adj1.T, 0, 1) 


def find_formation(home1,home2):
    Dtt=euclidean_distances(home1, home2)
    dendrogram = sch.dendrogram(sch.linkage(Dtt, method='ward'))
    model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    y_pred=model.fit_predict(Dtt)
    labels = model.labels_
    c_home = Counter(y_pred)
    plt.close()

    formation_home = Counter(sorted(c_home.elements())).values()
    s=""
    for i in formation_home:
        s=s+str(i)
        s=s+"-"
    s=s[:-1]

    return s
def transform(li):
    mean=np.mean(li)
    std=0
    su=0
    for i in li:
        j=(i-mean)**2
        su=su+j
    su=su/10
    su=su**(1/2)

    for i in range(0,len(li)):
        a=(li[i]-mean)/su
        li[i]=a
    return li

def plot_formation(ind,tracking_home,tracking_away,a_g,h_g):
    h_xs = tracking_home[[c for c in tracking_home.columns if 'Bris' in c and "ball" not in c and '_x' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and h_g != c[5:-2]]].iloc[ind].values
    h_ys = tracking_home[[c for c in tracking_home.columns if 'Bris' in c and "ball" not in c and '_y' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and h_g != c[5:-2]]].iloc[ind].values
    a_xs = tracking_away[[c for c in tracking_away.columns if 'Bris' not in c and "ball" not in c and '_x' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and a_g != c[5:-2]]].iloc[ind].values
    a_ys = tracking_away[[c for c in tracking_away.columns if 'Bris' not in c and "ball" not in c and '_y' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and a_g != c[5:-2]]].iloc[ind].values
    h_xs = h_xs[~np.isnan(h_xs)]
    a_xs = a_xs[~np.isnan(a_xs)]
    h_ys = h_ys[~np.isnan(h_ys)]
    a_ys = a_ys[~np.isnan(a_ys)]
    max_w,max_h=106,68
    ## Voronoi too included comment till line 42 if not needed"
    xs=np.concatenate((h_xs,a_xs)) 
    ys=np.concatenate((h_ys,a_ys)) 
    fig,ax = plot_pitch(field_color ='white')
    home_cord=[]
    for i in range(0,len(h_xs)):
        home_cord.append([h_xs[i], h_ys[i]])
    home_cord = np.array(home_cord) 
    away_cord=[]
    for i in range(0,len(a_xs)):
        away_cord.append([a_xs[i], a_ys[i]])
    away_cord = np.array(away_cord) 

    del_home = Delaunay(home_cord)
    del_away = Delaunay(away_cord)


    plt.triplot(home_cord[:,0], home_cord[:,1], del_home.simplices.copy())
    plt.plot(home_cord[:,0], home_cord[:,1], 'ro')
    plt.triplot(away_cord[:,0], away_cord[:,1], del_away.simplices.copy())
    plt.plot(away_cord[:,0], away_cord[:,1], 'bo')
    plt.xlim(-54,54)  #-54to54
    plt.ylim(-35,35)  #-35to35
    objs=ax.scatter(tracking_home.iloc[ind].ball_x, tracking_home.iloc[ind].ball_y, c='xkcd:grey', s=60.)
    """frame_minute =  int( tracking_home.iloc[ind]['Time [s]']/60. )
    frame_second =  ( tracking_home.iloc[ind]['Time [s]']/60. - frame_minute ) * 60.
    timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
    objs = ax.text(-2.5,max_h/2.+2., timestring, fontsize=14 )"""
    return fig,ax

def delanauy_triangle(ind,tracking_home,tracking_away,a_g,h_g):                                                     ### This c[5:-2] will work only if the team is Bris/Birm/Manu - 4 character
    h_xs = tracking_home[[c for c in tracking_home.columns if 'Bris' in c and "ball" not in c and '_x' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and h_g != c[5:-2]]].iloc[ind].values
    h_ys = tracking_home[[c for c in tracking_home.columns if 'Bris' in c and "ball" not in c and '_y' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and h_g != c[5:-2]]].iloc[ind].values
    a_xs = tracking_away[[c for c in tracking_away.columns if 'Bris' not in c and "ball" not in c and '_x' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and a_g != c[5:-2]]].iloc[ind].values
    a_ys = tracking_away[[c for c in tracking_away.columns if 'Bris' not in c and "ball" not in c and '_y' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and a_g != c[5:-2]]].iloc[ind].values
    h_xs = h_xs[~np.isnan(h_xs)]
    a_xs = a_xs[~np.isnan(a_xs)]
    h_ys = h_ys[~np.isnan(h_ys)]
    a_ys = a_ys[~np.isnan(a_ys)]

    home,away=get_dissim(h_xs,a_xs,h_ys,a_ys)
    return home,away

  

def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [img for img in os.listdir(pathIn) if img.endswith(".png")]
    print
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def convex_hull(ind,tracking_home,tracking_away,a_g,h_g):
    h_xs = tracking_home[[c for c in tracking_home.columns if 'Bris' in c and "ball" not in c and '_x' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and h_g != c[5:-2]]].iloc[ind].values
    h_ys = tracking_home[[c for c in tracking_home.columns if 'Bris' in c and "ball" not in c and '_y' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and h_g != c[5:-2]]].iloc[ind].values
    a_xs = tracking_away[[c for c in tracking_away.columns if 'Bris' not in c and "ball" not in c and '_x' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and a_g != c[5:-2]]].iloc[ind].values
    a_ys = tracking_away[[c for c in tracking_away.columns if 'Bris' not in c and "ball" not in c and '_y' in c and 'vx' not in c and 'vy' not in c and 'speed' not in c and a_g != c[5:-2]]].iloc[ind].values
    h_xs = h_xs[~np.isnan(h_xs)]
    a_xs = a_xs[~np.isnan(a_xs)]
    h_ys = h_ys[~np.isnan(h_ys)]
    a_ys = a_ys[~np.isnan(a_ys)]

    fig,ax = plot_pitch(field_color ='white')
    home_cord=[]
    for i in range(0,len(h_xs)):
        home_cord.append([h_xs[i], h_ys[i]])
    home_cord = np.array(home_cord) 
    
    away_cord=[]
    for i in range(0,len(a_xs)):
        away_cord.append([a_xs[i], a_ys[i]])
    away_cord = np.array(away_cord) 

    hull_home = ConvexHull(home_cord)
    hull_away = ConvexHull(away_cord)
    
    for simplex in hull_home.simplices:
        ax.plot(home_cord[simplex, 0], home_cord[simplex, 1], 'r--')
    #ax.plot(home_cord[hull_home.vertices,0], home_cord[hull_home.vertices,1], 'r--')
    #ax.plot(home_cord[hull_home.vertices[0],0], home_cord[hull_home.vertices[0],1], 'ro')
    cx = np.mean(hull_home.points[hull_home.vertices,0])
    cy = np.mean(hull_home.points[hull_home.vertices,1])

    plt.plot(cx, cy,'rx',ms=5)
    dist_home=[]
    for i in range(0,len(home_cord)):
        dist=(home_cord[i][0]-cx)*(home_cord[i][0]-cx)+(home_cord[i][1]-cy)*(home_cord[i][1]-cy)
        dist_home.append(sqrt(dist))

    h_st=mean(dist_home)#print("Home team stretch: ",mean(dist_home))

    for simplex in hull_away.simplices:
        ax.plot(away_cord[simplex, 0], away_cord[simplex, 1], 'b--')
        #x.append(away_cord[simplex, 0])
        #y.append(away_cord[simplex, 1])
    #ax.plot(away_cord[hull_away.vertices,0], away_cord[hull_away.vertices,1], 'b--')
    #ax.plot(away_cord[hull_away.vertices[0],0], away_cord[hull_away.vertices[0],1], 'bo')
    cx = np.mean(hull_away.points[hull_away.vertices,0])
    cy = np.mean(hull_away.points[hull_away.vertices,1])


    plt.plot(cx, cy,'bx',ms=5)
    dist_away=[]
    for i in range(0,len(away_cord)):
        dist=(away_cord[i][0]-cx)*(away_cord[i][0]-cx)+(away_cord[i][1]-cy)*(away_cord[i][1]-cy)
        dist_away.append(sqrt(dist))
    #print("Away team stretch: ",mean(dist_away))
    #plt.close(fig)
    a_st=mean(dist_away)
    return h_st, a_st

def FindPoint(x1, y1, x2, y2, x, y) : 
    if (x > x1 and x < x2 and 
        y > y1 and y < y2) : 
        return True
    else : 
        return False
    

### get the dataframe of tracking data for particular time e.g 1st half 34th minute 57 seconds
### For 25 seconds
def get_dataframe_particular_interval(full_match, period, minutes, seconds):
    if period>1:
        minutes = minutes - 45*(period-1)

    #minutes = minutes % 45
    total_secs = minutes*60 + seconds
    full_match_track = full_match[(full_match.period == period) & (full_match.sec <= (total_secs+2)) & (full_match.sec >= (total_secs-23))]   
    return full_match_track

### get the dataframe of tracking data for particular time e.g 1st half 34th minute 57 seconds
### For 25 seconds
def get_dataframe_particular_interval_kloppy(full_match, period, minutes, seconds):
    if period>1:
        minutes = minutes - 45*(period-1)

    #minutes = minutes % 45
    total_secs = minutes*60 + seconds
    full_match_track = full_match[(full_match.period == period) & (full_match.timestamp <= (total_secs+2)) & (full_match.timestamp >= (total_secs-23))]   
    return full_match_track

### get the dataframe of tracking data for particular time e.g 1st half 34th minute 57 seconds
### For 5 seconds

def get_dataframe_particular_interval_five_sec(full_match, period, minutes, seconds):
    if period>1:
        minutes = minutes - 45*(period-1)

    #minutes = minutes % 45
    total_secs = minutes*60 + seconds
    full_match_track = full_match[(full_match.period == period) & (full_match.timestamp <= (total_secs+3)) & (full_match.timestamp >= (total_secs-2))]   
    return full_match_track

def get_combined_dataframe(full_match, box_entries_df):
    full_match_track = pd.DataFrame()
    for ind, row in box_entries_df.iterrows():
        track_df = get_dataframe_particular_interval_five_sec(full_match, period = row["period"], minutes = row["minute"], seconds = row["second"]) 
        full_match_track = full_match_track.append(track_df)
    return full_match_track


def plot_pitchcontrol_for_event_from_laurie( event_id,  tracking_home, tracking_away, PPCF, brist_attack_side , alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (106.0,68), poss_team = "Bris", show_img = False, title = "Title", player_size = 15, teams = ["Bristol City", "Swansea"]):
    """ plot_pitchcontrol_for_event( event_id, events,  tracking_home, tracking_away, PPCF )
    
    Plots the pitch control surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
    
    Parameters
    ----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data 
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    NB: this function no longer requires xgrid and ygrid as an input
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """    

    # pick a pass at which to generate the pitch control surface
    pass_frame = event_id
    pass_team = poss_team

    period = tracking_home.iloc[event_id]['period']
    
    # plot frame and event
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    
    fig, ax = plot_frame(
        tracking_home.loc[pass_frame],
        tracking_away.loc[pass_frame],
        figax=(fig, ax),
        PlayerAlpha=alpha,
        include_player_velocities=include_player_velocities,
        annotate=annotate, 
        PlayerMarkerSize = player_size
    )    
    #plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
    
    # plot pitch control surface
    if pass_team=='Bris':
        cmap = 'bwr'
    else:
        cmap = 'bwr_r'
    im = ax.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)


    #### Code for defining the color bar
    cmap_bar = "bwr"
    if (brist_attack_side == "left" and period == 1) or (brist_attack_side == "right" and period == 2):
        cmap_bar = "bwr"
    else:
        cmap_bar = "bwr_r"


    ###### Change the vmax to vmin for displaying the color bar max and min colors change that in the set_ticks below too
    
    new_cmap_bar = truncate_colormap(plt.get_cmap(cmap_bar), 0.2, 0.8)
    sm = plt.cm.ScalarMappable(cmap = new_cmap_bar, norm = plt.Normalize(vmin=0, vmax = 1))
    cbar = plt.colorbar(sm, orientation = "horizontal") 

    if (brist_attack_side == "left" and period == 1) or (brist_attack_side == "right" and period == 2):
        cbar.set_ticks([1, 0])
    else:
        cbar.set_ticks([0, 1])
      
    cbar.set_ticklabels(["Bristol City", teams[1]])
    cbar.set_label('Pitch Control')

    frame_minute =  str( int(tracking_home.iloc[event_id]['minutes'] + 45 * (period-1)))
    frame_second =  str( int(tracking_home.iloc[event_id]['seconds']))
    if int(tracking_home.iloc[event_id]['seconds'])<10:
        frame_second = "0" + frame_second

    timestring = "%s:%s" % ( frame_minute, frame_second  )
    objs = ax.text(-5.5,36., timestring, fontsize=14 )
    
    if title != "Title":
        ax.set_title(title, y=1.08, fontsize = 20)
        
    ax.text(-62, -44, 
            "Pitch Control denotes the regions where the teams have control on the pitch\n i.e Regions where their players can reach the ball first if it is played over there.", 
            fontsize = 13)

    plt.xlim(-54,54) #-54to54
    plt.ylim(-35,35) #-35to35
    plt.savefig(image_folder + "output/PC/" + str(event_id) + ".png", dpi=150, facecolor = "white")
    

    
#     fig.canvas.draw()
#     data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    if show_img == False:
        plt.close(fig)
    return fig, ax

def plot_EPV_for_event( event_id, tracking_home, tracking_away, PPCF, EPV, poss_team = "Bris", alpha = 0.7, include_player_velocities=True, annotate=False, autoscale=0.1, contours=False, field_dimen = (106.0,68), show_img = False, title = "Title", player_size = 15, GK_numbers = ["1", "1"], opponent = "Card"):
    """ plot_EPV_for_event( event_id, events,  tracking_home, tracking_away, PPCF, EPV, alpha, include_player_velocities, annotate, autoscale, contours, field_dimen)
    
    Plots the EPVxPitchControl surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        EPV: Expected Possession Value surface. EPV is the probability that a possession will end with a goal given the current location of the ball. 
             The EPV surface is saved in the FoT github repo and can be loaded using Metrica_EPV.load_EPV_grid()
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        autoscale: If True, use the max of surface to define the colorscale of the image. If set to a value [0-1], uses this as the maximum of the color scale.
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """    

    # pick a pass at which to generate the pitch control surface
    pass_frame = event_id
    pass_team = poss_team

    period = tracking_home.iloc[event_id]['period']
    
    # plot frame and event
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    plot_frame( tracking_home.loc[pass_frame], tracking_away.loc[pass_frame], 
               figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, 
               annotate=annotate , PlayerMarkerSize = player_size)
    #plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
       
    brist_goalie = GK_numbers[0]
    opp_goalie = GK_numbers[1]
    # plot pitch control surface
    if pass_team=='Bris':
        cmap = 'Reds'
        lcolor = 'r'
        EPV = np.fliplr(EPV) if find_playing_direction(tracking_home, 'Bris', brist_goalie) == -1 else EPV
    else:
        cmap = 'Blues'
        lcolor = 'b'
        EPV = np.fliplr(EPV) if find_playing_direction(tracking_away, opponent, opp_goalie) == -1 else EPV
    
    EPVxPPCF = PPCF * EPV
    
    if autoscale is True:
        #vmax = np.max(EPVxPPCF)*2.

        ### If more darker regions at danger zones are required then uncomment the below line and commment previous one
        vmax = np.max(EPVxPPCF)
    elif autoscale>=0 and autoscale<=1:
        vmax = autoscale
    else:
        assert False, "'autoscale' must be either {True or between 0 and 1}"
        
    ax.imshow(np.flipud(EPVxPPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=vmax,cmap=cmap,alpha=0.7)
    
    if contours:
        ax.contour( EPVxPPCF,extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),levels=np.array([0.75])*np.max(EPVxPPCF),colors=lcolor,alpha=1.0)
    
    sm = plt.cm.ScalarMappable(cmap = cmap)
    cbar = plt.colorbar(sm, orientation = "horizontal")
    
    cbar.set_ticks([1, 0]) 
    cbar.set_ticklabels(["High", "Low"])
    
    cbar.set_label('Possession Value')

    frame_minute =  str( int(tracking_home.iloc[event_id]['minutes'] + 45 * (period-1)))
    frame_second =  str( int(tracking_home.iloc[event_id]['seconds']))
    
    if int(tracking_home.iloc[event_id]['seconds'])<10:
        frame_second = "0" + frame_second

    timestring = "%s:%s" % ( frame_minute, frame_second  )
    objs = ax.text(-5.5,36., timestring, fontsize=14 )
    
    if title != "Title":
        ax.set_title(title, y=1.08, fontsize = 20)
        
    ax.text(-65, -47, 
            "Possession value means the chances of scoring a goal in the next 3 actions. Higher the\npossession value, higher the chances of scoring. Darker regions are where there is high\nchances of goal happening in the next few actions if the ball is played there.", 
            fontsize = 12)

    plt.xlim(-54,54) #-54to54
    plt.ylim(-35,35) #-35to35
    plt.savefig(image_folder + "output/EPV/" + str(event_id) + ".png", dpi=150, facecolor = "white")
    
#     fig.canvas.draw()
#     data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    if show_img == False:
        plt.close(fig)

    return fig,ax

def plot_EPV(EPV,field_dimen=(106.0,68),attack_direction=1):
    """ plot_EPV( EPV,  field_dimen, attack_direction)
    
    Plots the pre-generated Expected Possession Value surface 
    
    Parameters
    -----------
        EPV: The 32x50 grid containing the EPV surface. EPV is the probability that a possession will end with a goal given the current location of the ball. 
             The EPV surface is saved in the FoT github repo and can be loaded using Metrica_EPV.load_EPV_grid()
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        attack_direction: Sets the attack direction (1: left->right, -1: right->left)
            
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """    
    if attack_direction==-1:
        # flip direction of grid if team is attacking right->left
        EPV = np.fliplr(EPV)
    ny,nx = EPV.shape
    # plot a pitch
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    # overlap the EPV surface
    ax.imshow(EPV, extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),vmin=0.0,vmax=0.6,cmap='Blues',alpha=0.6)
    



def find_playing_direction(team,teamname, goal_keeper):
    '''
    Find the direction of play for the team (based on where the goalkeepers are at kickoff). +1 is left->right and -1 is right->left
    '''    
    GK_column_x = teamname+"_"+str(goal_keeper)+"_x"
    # +ve is left->right, -ve is right->left
    return -np.sign(team.iloc[0][GK_column_x])
    
def find_goalkeeper(team):
    '''
    Find the goalkeeper in team, identifying him/her as the player closest to goal at kick off
    ''' 
    x_columns = [c for c in team.columns if c[-2:].lower()=='_x' and c[:4] in [match_params['0'], match_params['1']]]
    GK_col = team.iloc[0][x_columns].abs().idxmin(axis=1)
    return GK_col.split('_')[1]

def load_EPV_grid(fname = image_folder + 'EPV_grid.csv'):
    """ load_EPV_grid(fname='EPV_grid.csv')
    
    # load pregenerated EPV surface from file. 
    
    Parameters
    -----------
        fname: filename & path of EPV grid (default is 'EPV_grid.csv' in the curernt directory)
        
    Returns
    -----------
        EPV: The EPV surface (default is a (32,50) grid)
    
    """
    epv = np.loadtxt(fname, delimiter=',')
    return epv

def load_EPV_grid_df(fname = image_folder + 'EPV_grid_25_16.csv'):
    """ load_EPV_grid(fname='EPV_grid.csv')
    
    # load pregenerated EPV surface from file. 
    
    Parameters
    -----------
        fname: filename & path of EPV grid (default is 'EPV_grid.csv' in the curernt directory)
        
    Returns
    -----------
        EPV: The EPV surface (default is a (32,50) grid)
    
    """
    
    return pd.read_csv(fname).drop("Unnamed: 0", axis=1).to_numpy()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def check_offsides( attacking_players, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):
    """
    check_offsides( attacking_players, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):
    
    checks whetheer any of the attacking players are offside (allowing for a 'tol' margin of error). Offside players are removed from 
    the 'attacking_players' list and ignored in the pitch control calculation.
    
    Parameters
    -----------
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_position: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        verbose: if True, print a message each time a player is found to be offside
        tol: A tolerance parameter that allows a player to be very marginally offside (up to 'tol' m) without being flagged offside. Default: 0.2m
            
    Returrns
    -----------
        attacking_players: list of 'player' objects for the players on the attacking team with offside players removed
    """    
    # find jersey number of defending goalkeeper (just to establish attack direction)
    defending_GK_id = GK_numbers[1] if attacking_players[0].teamname=='Bris' else GK_numbers[0]
    # make sure defending goalkeeper is actually on the field!
    assert defending_GK_id in [p.id for p in defending_players], "Defending goalkeeper jersey number not found in defending players"
    # get goalkeeper player object
    defending_GK = [p for p in defending_players if p.id==defending_GK_id][0]  
    # use defending goalkeeper x position to figure out which half he is defending (-1: left goal, +1: right goal)
    defending_half = np.sign(defending_GK.position[0])
    # find the x-position of the second-deepest defeending player (including GK)
    second_deepest_defender_x = sorted( [defending_half*p.position[0] for p in defending_players], reverse=True )[1]
    # define offside line as being the maximum of second_deepest_defender_x, ball position and half-way line
    offside_line = max(second_deepest_defender_x,defending_half*ball_position[0],0.0)+tol
    # any attacking players with x-position greater than the offside line are offside
    if verbose:
        for p in attacking_players:
            if p.position[0]*defending_half>offside_line:
                print("player %s in %s team is offside" % (p.id, p.playername) )
    attacking_players = [p for p in attacking_players if p.position[0]*defending_half<=offside_line]
    return attacking_players



