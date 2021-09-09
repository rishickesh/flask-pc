#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
#sys.path.append("/Users/rishickesh/Downloads")

from tracking_utility import *
import pandas as pd
import matplotlib.pyplot as plt
from flask_session import Session
import matplotlib.patches as patches
from flask import Flask, send_file, make_response
import io
#from plot import do_plot


# In[2]:


## key value params for usage

### 1 - Home team, 0  - Away team

match_params = {'1' : "Card", 
                '0' : "Bris", 
                '10' : "ball"}

opponent = "Card"

track_params = {
    "Bris" : 0,  ### 0 if Bristol is away team
    "Card" : 1  #### 
}


home_away = {
    "1": "home",
    "0":"away"
}



teams = ["Bristol City", "Cardiff City"]

##included types 1 for home team , 0 for away, 10 for ball update this when checking the dataset
type_list = [0, 1, 10]

brist_goalie = str(1)  ### Bristol goalie jersey number
opp_goalie = str(1)

### Home team attack side in first half;  left: right->left ; right: left->right (decide this after seeing the PC graph)
brist_attack_side = "left"


### define the brist attacking side of the game in first half, left: right->left ; right: left->right
brist_side = "left"

GK_numbers = [brist_goalie, opp_goalie]


# In[3]:


full_match_track = pd.read_csv("Tracking_sample")
full_match_track


# In[4]:


def dataframe_team_split(full_match_track):
    cols_opp = [x for x in full_match_track.columns if "Bris" not in x]
    tracking_oppon = full_match_track[cols_opp]

    cols_opp = [x for x in full_match_track.columns if opponent not in x]
    tracking_brist = full_match_track[cols_opp]
    return tracking_brist, tracking_oppon
    


# In[5]:


tracking_brist, tracking_oppon = dataframe_team_split(full_match_track)
tracking_brist = calc_player_velocities(tracking_brist, smoothing=False, team_list = ["Bris", opponent])
tracking_oppon = calc_player_velocities(tracking_oppon, smoothing=False, team_list = ["Bris", opponent])


# In[6]:


def pitch_control(ind, tracking_attack,tracking_defend,params,field_dimen = (105.,68.,), poss_team = "Bris", def_team = 'Bren', GK_numbers = ["1", "1"]):
    #field_dimen = (105.,68.,) 
    
    ### or set it as 50 while loading the normal EPV
    n_grid_cells_x = 25 
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    xgrid = np.linspace( -field_dimen[0]/2, field_dimen[0]/2, n_grid_cells_x)    
    ygrid = np.linspace( -field_dimen[1]/2, field_dimen[1]/2, n_grid_cells_y )
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)))
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid))) 
    
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
    
    for i in range( len(ygrid) ):
        for j in range( len(xgrid)):
            target_position = np.array( [xgrid[j], ygrid[i]] )
            PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params)
          # check probabilitiy sums within convergence
    checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
    #assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    return PPCFa, xgrid, ygrid


# In[7]:


def plot_int(patches_li, ind):
    figw,axw= plot_pitch(field_color = "white")

    #ind = 50

    field_dimen = (106.0,68)
    period = brist_data.iloc[ind]['period']

    for team,color in zip( [brist_data.iloc[ind], oppon_data.iloc[ind]], ['r', 'b']) :
        x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
        y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y']

        for x, y in zip(x_columns, y_columns):
            pa = patches.Circle((team[x], team[y]), 1.4, fc = color, label = x[5:-2], alpha = 0.7)
            patches_li.append(pa)

        vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
        vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
        #print(pd.to_numeric(team[x_columns]) , type(team[x_columns]))
        #ax.quiver( pd.to_numeric(team[x_columns]), pd.to_numeric(team[y_columns]), pd.to_numeric(team[vx_columns]), pd.to_numeric(team[vy_columns]), color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=0.7)
        w=1
        [axw.text( team[x], team[y], x.split('_')[1], fontsize=6, color="w", ha="center", va="center") for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 

    pa = patches.Circle((brist_data.iloc[ind]['ball_x'], brist_data.iloc[ind]['ball_y']), 0.9, fc = "k", label = "ball", alpha = 0.7)
    patches_li.append(pa)
    
    pass_team = match_params["1"]
    
    
    def_team = match_params["0"]
    if brist_data.iloc[ind].ball_owning_team_id == "away":
        def_team = match_params["1"]
        pass_team = match_params["0"]

    PPCF, xgrid, ygrid = pitch_control(ind, brist_data, oppon_data, params, poss_team = pass_team, def_team = def_team, GK_numbers = GK_numbers)

    if pass_team=='Bris':
        cmap = 'bwr'
    else:
        cmap = 'bwr_r'
    im = axw.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)


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

    frame_minute =  str( int(brist_data.iloc[ind]['minutes'] + 45 * (period-1)))
    frame_second =  str( int(brist_data.iloc[ind]['seconds']))
    if int(brist_data.iloc[ind]['seconds'])<10:
        frame_second = "0" + frame_second

    timestring = "%s:%s" % ( frame_minute, frame_second  )
    objs = axw.text(-5.5,38., timestring, fontsize=12 )


    axw.set_title("Pitch Control", y=1.08, fontsize = 20)

    axw.text(-62, -46, 
            "Pitch Control denotes the regions where the teams have control on the pitch\n i.e Regions where their players can reach the ball first if it is played over there.", 
            fontsize = 13)

    return figw, axw, patches_li


# In[8]:



brist_data = tracking_brist.copy()
oppon_data = tracking_oppon.copy()

rows, cols = (16, 25)
arr = [[0.5]*cols]*rows

de = []

class DraggablePoint:
    lock = None #only one can be animated at a time
    def __init__(self, point):
        self.point = point
        self.press = None
        self.background = None
        
        self.x = None
        self.y = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        
        for t in axes.texts:
            self.x = t.get_position()[0]
            self.y = t.get_position()[1]
            if self.x == self.point.center[0] and self.y == self.point.center[1]:
                t.set_visible(False)
        
        self.point.set_animated(True)
        canvas.draw()
            
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        self.background = None
        

        ## Get color and label of dragged point
        color = self.point.get_fc()[0]
        color_2 = self.point.get_fc()[2]
        
        player_num = self.point.get_label()
        
        
        ###If its red / 1 change the Bristol dataframe or else toher
        if color == 1:
            team = "Bris"
            brist_data.at[ind, team + "_" + player_num + "_x"] = self.point.center[0]
            brist_data.at[ind, team + "_" + player_num + "_y"] = self.point.center[1]
        elif color_2 == 1:
            team = opponent
            oppon_data.at[ind, team + "_" + player_num + "_x"] = self.point.center[0]
            oppon_data.at[ind, team + "_" + player_num + "_y"] = self.point.center[1]
        else:
            brist_data.at[ind, "ball_x"] = self.point.center[0]
            brist_data.at[ind, "ball_y"] = self.point.center[1]
            oppon_data.at[ind, "ball_x"] = self.point.center[0]
            oppon_data.at[ind, "ball_y"] = self.point.center[1]
            
        
        field_dimen = (106.0,68)
        period = brist_data.iloc[ind]['period']
        
        pass_team = match_params["1"]
        def_team = match_params["0"]
        if brist_data.iloc[ind].ball_owning_team_id == "away":
            def_team = match_params["1"]
            pass_team = match_params["0"]

        PPCF, xgrid, ygrid = pitch_control(ind, brist_data, oppon_data, params, poss_team = pass_team, def_team = def_team, GK_numbers = GK_numbers)

        
        if pass_team=='Bris':
            cmap = 'bwr'
        else:
            cmap = 'bwr_r'
        
            
        im = self.point.axes.imshow(np.flipud(arr), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap, alpha = 1)

        im = self.point.axes.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap, alpha = 0.5)

        for team,color in zip( [brist_data.iloc[ind], oppon_data.iloc[ind]], ['r', 'b']) :
            x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
            y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y']

            
            [self.point.axes.text( team[x], team[y], x.split('_')[1], fontsize=7, color="w", ha="center", va="center") for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) and (team[x] == self.point.center[0] and team[y] == self.point.center[1]) ] 

        frame_minute =  str( int(brist_data.iloc[ind]['minutes'] + 45 * (period-1)))
        frame_second =  str( int(brist_data.iloc[ind]['seconds']))
        if int(brist_data.iloc[ind]['seconds'])<10:
            frame_second = "0" + frame_second

        timestring = "%s:%s" % ( frame_minute, frame_second  )
        self.point.axes.text(-5.5,38., timestring, fontsize=12 )
        
        self.point.axes.text(-62, -46, 
            "Pitch Control denotes the regions where the teams have control on the pitch\n i.e Regions where their players can reach the ball first if it is played over there.", 
            fontsize = 13)
        
        self.point.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)

        



# In[ ]:





# In[74]:

ind = 20
app1 = Flask(__name__)

@app.route('/matplot', methods=("POST", "GET"))
def mpl():
    return render_template('matplot.html',
                           PageTitle = "Matplotlib")
 


@app.route('/')
def plot_png():
          
    fig, ax, patches_li = plot_int([], ind)

    drs = []


    for circ in patches_li:
        ax.add_patch(circ)
        dr = DraggablePoint(circ)
        dr.connect()
        drs.append(dr)

    plt.show()
    
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    

Session(app1)


# In[ ]:





