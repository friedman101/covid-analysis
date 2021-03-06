#!/usr/bin/env python3

import numpy as np
import os.path
import os
import urllib.request
from urllib.parse import urlparse
import pandas as pd
import us
import matplotlib.pyplot as plt

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

def normalize_dataset(data):
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    return (data-min_val)/range_val

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

def plot_data(data):
    
    p = np.polyfit(data['density [people per km^2]'], data['deaths per thousand'], 2)
    data['adjusted deaths per thousand'] = data['deaths per thousand'] - np.polyval(p, data['density [people per km^2]'])
    data['state covid performance'] =  normalize_dataset(data['deaths per thousand'])
    data['adjusted state covid performance'] = normalize_dataset(data['adjusted deaths per thousand'])

    fig, ax = plt.subplots(1,1)

    x_vals = np.linspace(0, np.max(data['density [people per km^2]']), 1000)
    ax.plot(x_vals, np.polyval(p, x_vals), label='fit', color='black')
    for i in range(data.shape[0]):
        x = data['density [people per km^2]'].iloc[i]
        y = data['deaths per thousand'].iloc[i]
        ax.annotate(data['abbreviation'].iloc[i], (x,y))

    ax.set_xlabel('Density [People per km^2]')
    ax.set_ylabel('COVID-19 Deaths per Thousand')
    ax.set_title('COVID-19 Deaths vs Density')

    fig, ax = plt.subplots(1,1)

    data = data.sort_values(by='adjusted state covid performance')
    x_vals = list(range(data.shape[0]))
    for i in x_vals:
        v0 = data['state covid performance'].iloc[i]
        v1 = data['adjusted state covid performance'].iloc[i]
        ax.vlines(i, v0, v1, color='black')
        my_labels = ['nominal','adjusted'] if i == 0 else [None, None]
        ax.scatter(i, v0, s=80, facecolors='none', edgecolors='r', label=my_labels[0])
        ax.scatter(i, v1, s=80, facecolors='none', edgecolors='b', label=my_labels[1])
    ax.set_xticks(x_vals)
    ax.set_xticklabels(data['abbreviation'])
    ax.set_ylabel('Normalized Performance')
    ax.set_title('Normalized State COVID-19 Performance [higher is worse]')
    ax.legend()

    plt.tight_layout()
    

covid_csv_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
state_area_url = 'https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv'
state_population_url = 'https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv'

covid_df = get_data(covid_csv_url)
state_area_df = get_data(state_area_url)
state_population_df = get_data(state_population_url)

data = build_dataset(covid_df, state_area_df, state_population_df)
plot_data(data)
plt.show()



