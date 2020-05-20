import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import math
#open files
econ_data = pd.read_excel('Econ_Data_Nomis.xlsx', header = 9, 
                          names = ['Area', 'Weekly pay (gross)','err(wpg)', 
                                   'Weekly pay (basic)', 'err(wpb)', 
                                   'Hourly_pay (gross)', 'err(hpg)', 
                                   'Hourly_pay (basic)', 'err(hpb)'],
                          nrows = 371, skiprows = 254)
                          # dtype = {'Area': str, 'Weekly pay (gross)': float,'err(wpg)': float, 
                          #          'Weekly pay (basic)': float, 'err(wpb)': float, 
                          #          'Hourly_pay (gross)': float, 'err(hpg)': float, 
                          #          'Hourly_pay (basic)': float, 'err(hpb)': float})
def cut(x):
    return x[8:]

econ_data['Area'] = econ_data['Area'].apply(cut)
econ_data = econ_data.set_index('Area', drop = True)
econ_data = econ_data.drop('City of London')
econ_data["Weekly pay (gross)"] = pd.to_numeric(econ_data["Weekly pay (gross)"])
#print(econ_data["Weekly pay (gross)"])

covid_data = pd.read_excel('covid_deaths.xlsx', sheet_name = 'Table 2',
                           header = 4, usecols = [3,10,11,13,14], nrows = 381,
                           names = ['Area', 'Deaths', 'Rate', 'Lower CI', 'Upper CI'], 
                           index_col = 0)
covid_data = covid_data.drop(['Isles of Scilly', 'Isle of Anglesey'])
covid_data['Rate'] = pd.to_numeric(covid_data['Rate'])
print('len(covid)= ', len(covid_data))

pop_dens_data = pd.read_excel('population_estimates.xls', sheet_name = 'MYE 5',
                              header = 4, usecols = [1,2,5], 
                              names = ['Area', 'Area Type', 'People per sq. km'],
                              index_col = 0)
pop_dens_data = pd.concat(
    [pop_dens_data.loc[pop_dens_data['Area Type'] =='Unitary Authority'],
     pop_dens_data.loc[pop_dens_data['Area Type'] =='Metropolitan District'],
     pop_dens_data.loc[pop_dens_data['Area Type'] =='Non-metropolitan District'],
     pop_dens_data.loc[pop_dens_data['Area Type'] =='London Borough'],
     pop_dens_data.loc[pop_dens_data['Area Type'] =='Council Area'],
     pop_dens_data.loc[pop_dens_data['Area Type'] =='Local Government District']])
print('len(pop_dens)= ', len(pop_dens_data))
df = econ_data.merge(covid_data, how = 'inner', left_index = True, right_index = True)

df2 = pop_dens_data.merge(covid_data, how = 'inner', left_index = True, right_index = True)
regions = pd.read_csv('Region_Look_Up.csv', usecols = [2,4], 
                      skiprows = 1, names = ['Area', 'Region'], index_col = 0)
df2 = df2.merge(regions, how = 'outer', left_index = True, right_index = True)
df2['Region'] = df2['Region'].fillna('Wales')
df2 = (df2.dropna()
       .drop('Buckinghamshire'))

SE = df2[df2['Region']=='South East']
NW = df2[df2['Region']=='North West']
EM = df2[df2['Region']=='East Midlands']
EE = df2[df2['Region']=='East of England']
London = df2[df2['Region']=='London']
YH = df2[df2['Region']=='Yorkshire and The Humber']
SW = df2[df2['Region']=='South West']
WM = df2[df2['Region']=='West Midlands']
Wales = SE = df2[df2['Region']=='Wales']
NE = df2[df2['Region']=='North East']
N = pd.concat([NE, NW])
M = pd.concat([EM, WM])
S = pd.concat([SE,SW])
#df2.set_index('Area Type', append = True)
unitary_authorities = df2[df2['Area Type']=='Unitary Authority']
metropolitan_districts = df2[df2['Area Type']=='Metropolitan District']
non_metropolitan_districts = df2[df2['Area Type']=='Non-metropolitan District']
london_boroughs = df2[df2['Area Type']=='London Borough'] 


def onclick(event):
    #plt.cla()
    global figtext
    figtext.remove()
    dists = []
    for i in range(len(df2)):
        xi = df2.iloc[i]['People per sq. km']
        yi = df2.iloc[i]['Rate']
        xmax = df2['People per sq. km'].max()
        ymax = df2['Rate'].max()
        x = (event.xdata - xi)/ xmax
        y = (event.ydata - yi)/ymax
        dists += [np.sqrt(x**2 + y**2)]
    index = dists.index(min(dists))
    txt = df2.iloc[index].name
    figtext = plt.annotate(txt, (df2.iloc[index]['People per sq. km']+5, 
                                 df2.iloc[index]['Rate']))
    plt.xlabel('People per sq. km')
    plt.ylabel('Covid-19 deaths per 100,000 people')
    plt.show()
    plt.draw()
        
plt.close('all')
plt.figure()
# plt.scatter(x=unitary_authorities['People per sq. km'], 
#             y=unitary_authorities['Rate'], 
#             alpha = 0.5, label = 'Unitary Authorities')
# plt.scatter(x=metropolitan_districts['People per sq. km'], 
#             y=metropolitan_districts['Rate'], 
#             alpha = 0.5, label = 'Metropolitan Districts')
# plt.scatter(x=london_boroughs['People per sq. km'], 
#             y=london_boroughs['Rate'], 
#             alpha = 0.5, label = 'London Boroughs')
# plt.scatter(x=non_metropolitan_districts['People per sq. km'], 
#             y=non_metropolitan_districts['Rate'], 
#             alpha = 0.5, label = 'Non-metropolitan districts')
regions = [S, N, EE, M, London, YH, Wales]
region_names = ['South of England', 'North of England', 'East of England', 
                'Midlands', 'London', 'Yorkshire and the Humber', 'Wales']
cmap = cm.get_cmap('Set1')
colors = cmap.colors
for index in range(len(regions)):
    i = regions[index]
    j = region_names[index]
    plt.scatter(x=i['People per sq. km'], 
                y=i['Rate'], 
                alpha = .6, label = j, color = colors[index], edgecolor = colors[index])    
plt.title('Covid 19 death rates in England and Wales')
plt.legend()
plt.xlabel('People per sq. km')
plt.ylabel('Covid-19 deaths per 100,000 people')
# unitary_authorities.plot.scatter(x ='People per sq. km', y='Rate', alpha = 0.5)
# council_areas.plot.scatter(x ='People per sq. km', y='Rate', alpha = 0.5)
# metropolitan_districts.plot.scatter(x ='People per sq. km', y='Rate', alpha = 0.5)
# non_metropolitan_districts.plot.scatter(x ='People per sq. km', y='Rate', alpha = 0.5)

figtext = plt.figtext(0.13,0.84, "Click on a point to see district name")
plt.gcf().canvas.mpl_connect('button_press_event', onclick)