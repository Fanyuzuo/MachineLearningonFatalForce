%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
from geopy.geocoders import Nominatim
import geopy

matplotlib.style.use('ggplot')

d = pd.read_csv("shooting_data_with_county_covariates.csv", encoding='latin-1')
d[['Latitude', 'Longitude']] = d[['lat', 'lon']].fillna(0)
#d.shape

d['Total victims'] = 1
df = pd.DataFrame(d.year)
df = pd.DataFrame(df.year).join(d['Total victims'])
df = pd.DataFrame(df.groupby('year', as_index=True).sum())
#df
df.plot(kind = 'bar', figsize=(10,9), color='r')



#Build the rating of total victims of shooting by state.

#Lets convert locations on States and clean up it
states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'D.C': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

vd = pd.DataFrame(d.state)
vd['State'] = vd['state']
vd = vd.join(d['Total victims'])


for state in vd.State.values:
    try:
        if state in states.keys():
            vd.State.replace({state: states[state]}, inplace=True)
        elif type(int(state)) == int:
            vd.State.replace({state: 'Kentucky'}, inplace=True)
        else:
            pass
    except: ValueError

vd = pd.DataFrame(vd.groupby('State', as_index=False).sum())
vd = vd.sort_values(by='Total victims', ascending=False)
#vd.info()
vd.plot(kind = 'bar', x='State', y='Total victims', figsize=(15,12), color='b')



#Set the coordinates instead NaN values

md = pd.DataFrame(d.state)
md = md.join(d[['Latitude', 'Longitude', 'Total victims']])
zero_c = md[md.Latitude == 0]

#md.info()


#Visualize mass shootings on US map

from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
us_map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

us_map.drawcoastlines() #zorder=3
us_map.drawmapboundary(zorder=0) #fill_color='#9fdbff'
us_map.fillcontinents(color='#ffffff',zorder=1) #,lake_color='#9fdbff',alpha=1
us_map.drawcountries(linewidth=1.5, color='darkblue') #color='darkblue'
us_map.drawstates(zorder=3) #zorder=3
#plt.show()

#Set county location values, shooting level values, marker sizes (according to county size), colormap and title 
x, y = us_map(md.Longitude.tolist(), md.Latitude.tolist())
colors = (md['Total victims']).tolist()
sizes = (md['Total victims']*2).tolist()
cmap = plt.cm.YlOrRd
sm = ScalarMappable(cmap=cmap)
plt.title('US shooting victims')

scatter = ax.scatter(x,y,s=sizes,c='r',cmap=cmap,alpha=1,edgecolors='face',marker='o',zorder=3)
plt.show()



#run separately

import seaborn as sns

test_table = pd.DataFrame(d[['gender', 'race']], columns=['gender', 'race'])
test_table.gender.replace({'M': 'Male', 'F': 'Female'}, inplace=True)
test_table.race.replace({'W': 'White', 'B': 'Black or African American', 'H' : 'Hispanic and Latino Americans', 'O': 'Other Race', 'A': 'Asian', 'N': 'Native American'}, inplace=True)
gender_table = pd.crosstab(test_table['race'], test_table['gender'])
gender_table

#run separately
race_table =pd.crosstab(test_table['gender'], test_table['race'])
corr = race_table.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

#run separately

corr = gender_table.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
sns.plt.suptitle('Correlation between gender')

