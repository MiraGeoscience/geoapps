import ipywidgets as widgets
from ipywidgets import interactive, interact
from IPython.display import clear_output
from IPython.display import display
from ipywidgets import IntProgress
from ipywidgets import Dropdown

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import math
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster
import statsmodels.formula.api as sm
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage, centroid
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as signal
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import centroid, fcluster
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as shc

df_imp = pd.read_csv(r"C:\Users\jeanphilippep\OneDrive - mirageoscience\PROJECTS\API_Python\Litho_Classif\data\ASCII_COLUMN\geochem.csv")

#Get objects from GA with dro down to select which object
#convert to df (name df_imp)

dropdown = widgets.SelectMultiple(
                        options=list(df_imp.columns.values),
                        description='Variables:',
                        disabled=False,
                        layout={'height':'100px', 'width':'40%'},
                        continous_update=False)

def make_input_map():
    maxg = len(selection)
    maxr = math.ceil((len(selection)/2))
    maxg_arg=list()
    for i in range(0, maxr):
        maxg_arg.append([{},{}])     
    ic = 1
    ir = 1
    fig = make_subplots(rows=maxr, cols=2, specs=maxg_arg, print_grid=False,
                       subplot_titles=selection)
    for s in selection:
        if ic >2:
            ic=1
        fig.add_trace(go.Scatter(x=df['X'], y=df['Y'], mode='markers', name=s, 
                                 marker=dict(size=3,
                                 color=((df[s]-min(df[s]))/(max(df[s])-min(df[s]))), #set color equal to a variable
                                 colorscale='Rainbow', # one of plotly colorscales
                                 showscale=True
                                    ),
                                text=df[s]),row=ir, col=ic)
        if ic == 2:
            ir=ir+1
        ic=ic+1
    fig.update_yaxes(title_text="Y", showticklabels=False)
    fig.update_xaxes(title_text="X", showticklabels=False)
    fig.update_layout(title='Data Maps (0-1 color scaled)', width=950, height=300*maxr, showlegend=False)
    fig.show()

def make_correl_map():
    corrs = filtered_df.corr()
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x = list(corrs.columns), y = list(corrs.index), z = corrs.values, 
                             type = 'heatmap', colorscale = 'Viridis', zsmooth='fast'))
    fig.update_layout(width=500, height=500, autosize=False, margin=dict(t=0, b=0, l=0, r=0),template="plotly_white",)
    fig.update_scenes(aspectratio=dict(x=1, y=1, z=0.7), aspectmode="manual")
    fig.update_layout( updatemenus=[
        dict(buttons=list([
        dict(args=["type", "heatmap"], label="Heatmap", method="restyle"),
        dict(args=["type", "surface"], label="3D Surface", method="restyle"),]),
        direction="down", pad={"r": 10, "t": 10}, showactive=True,  x=0.01, xanchor="left", y=1.15, yanchor="top"),
        dict(buttons=list([
        dict(args=["colorscale", "Viridis"], label="Viridis", method="restyle"),
        dict(args=["colorscale", "Rainbow"], label="Rainbow", method="restyle"),
        dict(args=["colorscale", "Cividis"], label="Cividis", method="restyle"),
        dict(args=["colorscale", "Blues"], label="Blues", method="restyle"),
        dict(args=["colorscale", "Greens"], label="Greens", method="restyle"),]),
        direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.32, xanchor="left", y=1.15, yanchor="top")])
    fig.update_yaxes(autorange="reversed")
    fig.show()
    


def filter_dataframe(widget):
    global filtered_df, df, selection
    selection = list(widget['new'])
            
    with out:
        clear_output()
        df=df_imp.copy()
        for s in selection:
            df.drop(df[df[s] <= 0].index, inplace=True)
        filtered_df = df.loc[:,selection]
        display(filtered_df.describe(percentiles=None, include=None, exclude=None))

        if len(filtered_df)>0:
            make_correl_map()         
            make_input_map()
            
out = widgets.Output()
dropdown.observe(filter_dataframe, names='value')
display(dropdown)
display(out)

names = filtered_df.columns

scaled_df = filtered_df
scaling = [1] * len(names)
scaling_dict = dict(zip(names,scaling))

print('Adjust scaling of the data to represent feature importance')

def make_histo():
    maxg = len(names)
    maxr = math.ceil((len(names)/2))
    maxg_arg=list()
    for i in range(0, maxr):
        maxg_arg.append([{},{}])     
    ic = 1
    ir = 1
    fig = make_subplots(rows=maxr, cols=2, specs=maxg_arg, print_grid=False, subplot_titles=names)
    for n in names:
        if ic >2:
            ic=1
        fig.add_trace(go.Histogram(x=scaled_df[n], histnorm='percent', name=n),row=ir, col=ic)
        if ic == 2:
            ir=ir+1
        ic=ic+1
    
    fig.update_layout(title='Data Distributions', width=950, height=300*maxr, showlegend=False)
    fig.show()

for n in names:
      
    scaler = 'scale'+str(n) 
    out_scale = 'out_s'+str(n)
    out_scale = widgets.Output()
        
    scaler = widgets.IntSlider(min=1, max=10, step=1, value=1, description='Scale '+n, continous_update=False)
    
    display(scaler,out_scale)

    def scale_data(changesc):
        
        with out_scale:            
            clear_output()
            newV =(changesc['owner'].value)
            owner = (changesc['owner'].description)   
            scaling_dict[str(owner.replace('Scale ',''))] = str(newV)
                        
            for k in scaling_dict:
                s=float(scaling_dict[k])
                scaled_df[k] = ((scaled_df[k]-min(scaled_df[k]))/(max(scaled_df[k])-min(scaled_df[k])))*s
            
            make_histo()
        return changesc
    
    scaler.observe(scale_data, names ='value')

#WOULD BE COOL TO HAVE PROGRESS BAR HERE

Z = linkage(scaled_df, method='ward', metric='euclidean', optimal_ordering=False)
tsne = TSNE() 
tsne_results = tsne.fit_transform(scaled_df)

MaxK=30

clustersperf = fcluster(Z, MaxK, criterion='maxclust')

last = Z[-MaxK:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]

sc_slider = widgets.IntSlider(min=1, max=MaxK, step=1, value=1, description='Clusters:', continous_update=False)
output = widgets.Output()

buttonC = widgets.Button(description="Make Tree")
outputC = widgets.Output()

display(sc_slider, output)
display(buttonC, outputC)

def clusters(b):  
    global nbrC, c_dist 
    
    with output:
        clear_output()
        
        nbrC = b['owner'].value
        c_dist = last_rev[np.where(idxs == (nbrC-1))]
        print('Number of Clusters: ', nbrC)
        
        fig = make_subplots(rows=2, cols=2, 
                    specs=[[{}, {"rowspan": 2}],
                    [{}, None],
                    ],
                    print_grid=False,
                    subplot_titles=("Cluster Distance", "t-SNE Plot", "Acceleration", "Dendrogram"))
        
        fig.add_trace(go.Scatter(x=idxs, y=last_rev, mode='lines'), row=1, col=1)
        fig.add_shape(dict( type="line", x0=nbrC, y0=min(acceleration_rev), x1=nbrC, y1=max(last_rev), 
                           line=dict(color="Red", width=3,dash="dot")), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=idxs, y=acceleration_rev, mode='lines'), row=2, col=1)
        fig.add_shape(dict( type="line", x0=nbrC, y0=min(acceleration_rev), x1=nbrC, y1=max(acceleration_rev), 
                           line=dict(color="Red", width=3,dash="dot")), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=tsne_results[:,0], y=tsne_results[:,1], mode='markers'), row=1, col=2)
                  
        fig.update_layout(showlegend=False, title_text='Clusters Performance')
        fig.show() 



def on_button_c(c):
    global tsne_df
    with output:
        
        fig = ff.create_dendrogram(scaled_df, orientation='bottom', linkagefun=lambda x: Z, color_threshold =c_dist) 
        
        fig.add_shape(dict( type="line", y0=c_dist, y1=c_dist, 
                           line=dict(color="Red", width=3,dash="dot")))
        
        fig.update_layout({'width':950, 'height':600, 'title': 'Distance Dendrogram'})
        fig.update_xaxes(ticks = "", showgrid=False, showticklabels=False)
        fig.show()
         
        df['clust'] = fcluster(Z, nbrC, criterion='maxclust')
        df["clust"] = df["clust"].astype(str)
        df["clustn"] = df["clust"].astype(int)
        #filtered_df['clust'] = fcluster(Z, sc_slider.value, criterion='maxclust')
                 
        tsne_df = pd.DataFrame({'tsneX': tsne_results[:, 0], 'tsneY': tsne_results[:, 1], 'clust': np.array(df.clust)})


buttonC.on_click(on_button_c)       
sc_slider.observe(clusters)


clust_color = list(df.clust.unique())
clust_color.sort()
clust_dict = ['black'] * len(clust_color)
colors_map = dict(zip(clust_color, clust_dict))

clust_colorn = list(df.clustn.unique())
clust_colorn.sort()
colors_mapn = dict(zip(clust_colorn, clust_dict))

for i in clust_color:

    cp = 'colorpicker'+str(i) 
    output = 'out'+str(i)
    output = widgets.Output()
    
    cp = widgets.ColorPicker(
            concise=False,
            description=('Cluster '+str(i)),
            value='black',
            disabled=False
            )
    display(cp,output)
    
    def assign_color(change):
        
        with output:
            clear_output()
            newC =(change['owner'].value)
            ownerC = (change['owner'].description)   
            colors_map[str(ownerC.replace('Cluster ', ''))] = str(newC)
            colors_mapn[int(ownerC.replace('Cluster ', ''))] = str(newC)
            clust_dict.append(change['new'])
                        
            fig=px.scatter(tsne_df, x='tsneX', y='tsneY', color='clust', hover_name='clust', color_discrete_map=colors_map) 
            fig.update_layout({'title': 't-SNE Clustered Data Map'})
            fig.show()
            
    cp.observe(assign_color, names ='value')
    
groups = list(df.clust.unique())
groups.sort()
maxg = len(names)+2
maxr = math.ceil((len(names)/2))
colors_mapnl = list(colors_mapn.values())
maxg_arg=list([[{"rowspan": 2, "colspan": 2}, None], [None, None]])

for i in range(0, maxr):
    maxg_arg.append([{},{}])
fig = make_subplots(
    rows=2+maxr, cols=2,
    specs=maxg_arg,
    print_grid=False,
    subplot_titles=['Map', *names])       
for g in groups:
    fig.add_trace(go.Scatter(x=df.loc[df['clust'] == g].X, y=df.loc[df['clust'] == g].Y, mode='markers',
                        marker = dict(size=5, color = colors_map.get(g)), name='cluster '+g, showlegend=True),
                  row=1, col=1)
ic = 1
ir = 3
for n in names:        
    if ic >2:
        ic=1    
    for g in groups:
        fig.add_trace(go.Box(x=df.loc[df['clust'] == g].clust, y=df.loc[df['clust'] == g,n], 
                             fillcolor=colors_map.get(g), marker_color=colors_map.get(g), line_color=colors_map.get(g),
                             name=g, showlegend=False), row=ir, col=ic)    
    if ic == 2:
        ir=ir+1
    ic=ic+1
fig.update_layout(width=950, height=400*maxr,title='Clustered Data Map', showlegend=True)
fig.show()

