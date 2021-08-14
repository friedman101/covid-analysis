#!/usr/bin/env python

import numpy as np
import os.path
import urllib.request
import pandas as pd


covid_csv = 'us-states.csv'
covid_csv_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'

if not os.path.isfile(covid_csv):
    urllib.request.urlretrieve(covid_csv_url, covid_csv)


covid_df = pd.read_csv(covid_csv)
print(covid_df)