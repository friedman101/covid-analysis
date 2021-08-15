#!/usr/bin/env python3

import numpy as np
import os.path
import os
import urllib.request
from urllib.parse import urlparse
import pandas as pd
import us
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

def get_state_name(state_abbr):
    return us.states.lookup(state_abbr).name

def get_state_abbr(state_name):
    return us.states.lookup(state_name).abbr

def get_file(url):
    a = urlparse(url)
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    file = data_dir + '/' + os.path.basename(a.path)
    if not os.path.isfile(file):
        urllib.request.urlretrieve(url, file)
    return file

def get_data(url):
    return pd.read_csv(get_file(url))

def build_dataset(covid_df, state_area_df, state_population_df):
    states = set(covid_df['state'])

    covid_dict = {}
    for i in range(covid_df.shape[0])[::-1]:
        state = covid_df.iloc[i]['state']
        if len(covid_dict.keys()) == len(states):
            break
        if state not in covid_dict:
            covid_dict[state] = covid_df.iloc[i]['deaths']

    state_area_dict = dict(state_area_df.values)
    state_population_dict = {}
    for i in range(state_population_df.shape[0]):
        state_abbr = state_population_df['state/region'].iloc[i]
        if state_abbr == 'USA':
            continue;
        if state_population_df['ages'].iloc[i] != 'total' or state_population_df['year'].iloc[i] != 2013:
            continue
        state = get_state_name(state_abbr)
        state_population_dict[state] = state_population_df['population'].iloc[i]

    all_dict = {}
    sq_mi_to_sq_km = 2.58999
    for state in states:
        if state in covid_dict and state in state_population_dict and state in state_area_dict:
            all_dict[state] = {}
            all_dict[state]['covid deaths'] = covid_dict[state]
            all_dict[state]['population'] = state_population_dict[state]
            all_dict[state]['area [km^2]'] = state_area_dict[state]*sq_mi_to_sq_km
            all_dict[state]['abbreviation'] = get_state_abbr(state)

    data = pd.DataFrame.from_dict(all_dict, orient='index')
    data['density [people per km^2]'] = data['population']/data['area [km^2]']
    data['deaths per thousand'] = data['covid deaths']/data['population']*1000

    states_to_remove = ['District of Columbia', 'Puerto Rico']
    data = data.drop(states_to_remove)

    return data

def plot_map(data, ax, metric, title):

    plt.sca(ax)

    m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
    m.readshapefile('data/st99_d00', name='states', drawbounds=True)

    state_names = []
    colors = {}
    colors_adjusted = {}
    cmap = plt.cm.YlOrRd
    for shape_dict in m.states_info:
        state_name = shape_dict['NAME']
        state_names += [state_name]
        if state_name in ['District of Columbia','Puerto Rico']:
            continue
        
        min_val = np.min(data[metric])
        max_val = np.max(data[metric])
        range_val = max_val-min_val
        metric_val = (data.at[state_name,metric]-min_val)/range_val
        colors[state_name] = cmap(metric_val)[:3]

    ax = plt.gca() # get current axes instance
    for nshape,seg in enumerate(m.states):
        if state_names[nshape] in ['District of Columbia','Puerto Rico']:
            continue

        color = rgb2hex(colors[state_names[nshape]]) 
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)

    ax.set_title(title)
        


def plot_data(data):
    
    p = np.polyfit(data['density [people per km^2]'], data['deaths per thousand'], 2)
    data['adjusted deaths per thousand'] = data['deaths per thousand'] - np.polyval(p, data['density [people per km^2]'])

    fig, ax = plt.subplots(1,1)

    ax.scatter(data['density [people per km^2]'], data['deaths per thousand'], label='data')
    x_vals = np.linspace(0, np.max(data['density [people per km^2]']), 1000)
    ax.plot(x_vals, np.polyval(p, x_vals), label='fit', color='black')
    for i in range(data.shape[0]):
        x = data['density [people per km^2]'].iloc[i]
        y = data['deaths per thousand'].iloc[i]
        ax.annotate(data['abbreviation'].iloc[i], (x,y))

    ax.set_xlabel('Density [People per km^2]')
    ax.set_ylabel('Covid Deaths per Thousand')
    ax.set_title('Covid Deaths vs Density')
    ax.legend()

    fig, ax = plt.subplots(2,1)

    data = data.sort_values(by='deaths per thousand')
    x_vals = list(range(data.shape[0]))
    ax[0].bar(x_vals, data['deaths per thousand'])
    ax[0].set_xticks(x_vals)
    ax[0].set_xticklabels(data['abbreviation'])
    ax[0].set_ylabel('Deaths per Thousand')
    ax[0].set_title('Covid Deaths per Thousands')

    data = data.sort_values(by='adjusted deaths per thousand')
    ax[1].bar(x_vals, data['adjusted deaths per thousand'])
    ax[1].set_xticks(x_vals)
    ax[1].set_xticklabels(data['abbreviation'])
    ax[1].set_ylabel('Covid Deaths per Thousand')
    ax[1].set_title('Adjusted Deaths per Thousands')

    plt.tight_layout()
    

covid_csv_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
state_area_url = 'https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv'
state_population_url = 'https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv'
us_state_shapefile_urls = ['https://github.com/matplotlib/basemap/raw/master/examples/st99_d00.shp',
                        'https://github.com/matplotlib/basemap/raw/master/examples/st99_d00.shx',
                        'https://github.com/matplotlib/basemap/raw/master/examples/st99_d00.dbf']

covid_df = get_data(covid_csv_url)
state_area_df = get_data(state_area_url)
state_population_df = get_data(state_population_url)
for url in us_state_shapefile_urls:
    get_file(url)

data = build_dataset(covid_df, state_area_df, state_population_df)
plot_data(data)

fig, ax = plt.subplots(2,1)
plot_map(data, ax[0], 'deaths per thousand', 'Deaths per Thousand')
plot_map(data, ax[1], 'adjusted deaths per thousand', 'Adjusted Deaths per Thousand')
plt.show()



