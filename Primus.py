""" This file is my analysis on whether the Presidio/Presidio Heights
	is the safest neighborhood in SF. Also an exploration of what 
	types of crime happen in the Presidio.
"""
__version__ = '1.0'
__author__ = 'Kripa Dharan'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pandas imports the CSV as a Data Frame object
sfcrime2018 = pd.read_csv("SFCrime__2018_to_Present.csv")
sfcrimepast = pd.read_csv("SFCrime_2003_to_May_2018.csv")
neighborhood_areas = pd.read_csv("SF_neighborhood_areas.csv")
analysisN = pd.read_csv("SF neighborhoods - Sheet2.csv", index_col='Neighborhood')

#MAKE INDEXES INCIDENT CATEGORY AND COLUMNS ANALYSIS NEIGHBORHOOD
crime_catdist = pd.crosstab(index=sfcrime2018["Incident Category"],
                            columns=sfcrime2018["Analysis Neighborhood"])
crime_catpres = crime_catdist[['Presidio', 'Presidio Heights']]
analysisN.info()

#MAKE THE INDEXES THE ANALYSIS NEIGHBORHOODS AND THE COLUMNNS INCIDENT CATEGORY
crime_distcat = pd.crosstab(index=sfcrime2018["Analysis Neighborhood"],
                            columns=sfcrime2018["Incident Category"])
#DROP THE NEIGHBORHOODS THAT DONT HAVE POPULATION DATA
drop = ['Golden Gate Park', 'Hayes Valley', 'Japantown', 'Lincoln Park', 'Lone Mountain/USF', 'McLaren Park', 'Portola']
crime_distcat = crime_distcat.drop(drop)

#CREATE A COLOR COLUMN TO ALLOW FOR CUSTOM COLORING
crime_distcat['Color'] = 'b'
crime_distcat['Total Crime'] = crime_distcat.sum(axis = 1)

#TOTAL CRIMES BY DISTRICT(NOT PER CAPITA)
crime_sorted = crime_distcat.sort_values('Total Crime', ascending = True)
crime_sorted.loc['Presidio', 'Color'] = 'g'
crime_sorted.loc['Presidio Heights', 'Color'] = 'g'
crime_sorted.plot(y = 'Total Crime', kind = 'bar', color = crime_sorted['Color'])
plt.title('Total Crime by District')
plt.ylabel('Number of Crimes')
plt.show()

#COMBINE THE NEIGHBORHOOD POPULATION DATASET WITH MY CRIME_DISTCAT CROSSTAB
crime_distcat = pd.concat([crime_distcat, analysisN], axis=1)
crime_distcat['Per_capita Assault'] = crime_distcat['Assault'] / crime_distcat['Population']
crime_distcat['Dangerous crime'] = crime_distcat['Assault'] + crime_distcat['Arson'] + crime_distcat['Homicide'] + crime_distcat['Human Trafficking, Commercial Sex Acts'] + crime_distcat['Larceny Theft'] + crime_distcat['Motor Vehicle Theft'] + crime_distcat['Rape'] + crime_distcat['Robbery'] + crime_distcat['Sex Offense'] + crime_distcat['Stolen Property'] + crime_distcat['Weapons Offence'] + crime_distcat['Weapons Offense']
crime_distcat['Per_capita Dangerous crime'] = crime_distcat['Dangerous crime'] / crime_distcat['Population']

#MOVE THE AREAS FROM EHT AREAS DATASET TO AN ADDITIONAL COLUMN IN THE CRIME_DISTCAT DATASET
crime_distcat['Area'] = 0
for row in crime_distcat.index.values:
	crime_distcat.set_value(row, 'Area', neighborhood_areas[neighborhood_areas['Neighborhood'] == row]['Area']) 	
crime_distcat['Per_area Dangerous crime'] = crime_distcat['Dangerous crime'] / crime_distcat['Area']

#CALCULATE TOTAL CRIME PER CAPITA AND PER AREA
crime_distcat['Total Crime per_capita'] = crime_distcat['Total Crime'] / crime_distcat['Population']
crime_distcat['Total Crime per_area'] = crime_distcat['Total Crime'] / crime_distcat['Area']



#PER CAPITA DANGEROUS CRIME WITHOUT FINANCIAL DISTRICT
crime_distcat_nofin = crime_distcat.drop('Financial District/South Beach')
crime_distcat_nofin_sorted = crime_distcat_nofin.sort_values('Per_capita Dangerous crime', ascending = True)
crime_distcat_nofin_sorted.loc['Presidio','Color'] = 'g'
crime_distcat_nofin_sorted.loc['Presidio Heights', 'Color'] = 'g'
crime_distcat_nofin_sorted.plot(y = 'Per_capita Dangerous crime', kind = 'bar', color = crime_distcat_nofin_sorted['Color'])
plt.title('Dangerous Crime Per Capita by Neighborhood (without Financial District)')
plt.ylabel('Number of Crimes Per Capita')
plt.show()


#PER CAPITA DANGEROUS CRIME
crime_distcat.sort_values('Per_capita Dangerous crime', ascending = True).plot(y = 'Per_capita Dangerous crime', kind = 'bar', color = 'b')
plt.title('Dangerous Crime Per Capita by Neighborhood')
plt.ylabel('Number of Crimes Per Capita')
plt.show()


#MOST COMMON CRIMES IN THE PRESIDIO
crime_catdist['Presidio Percent'] = crime_catdist['Presidio'] / crime_distcat.loc['Presidio', 'Total Crime']
crime_catdist.sort_values('Presidio Percent', ascending = True)[-10:].plot(y = 'Presidio Percent', kind = 'bar')
plt.title('Most Common Crimes in the Presidio')
plt.ylabel('Percentage of Total Crime')
plt.show()

#MOST COMMON CRIMES IN PRESIDIO HEIGHTS
crime_catdist['Presidio Heights Percent'] = crime_catdist['Presidio Heights'] / crime_distcat.loc['Presidio Heights', 'Total Crime']
crime_catdist.sort_values('Presidio Heights Percent', ascending = True)[-10:].plot(y = 'Presidio Heights Percent', kind = 'bar')
plt.title('Most Common Crimes in Presidio Heights')
plt.ylabel('Percentage of Total Crime')
plt.show()


#DANGEROUS CRIME BY AREA
d_sorted_by_area = crime_distcat.sort_values('Per_area Dangerous crime', ascending = True)
d_sorted_by_area.loc['Presidio', 'Color'] = 'g'
d_sorted_by_area.loc['Presidio Heights', 'Color'] = 'g'
d_sorted_by_area.plot(y = 'Per_area Dangerous crime', kind = 'bar', color = d_sorted_by_area['Color'])
plt.title('Number of Dangerous Crimes by Area')
plt.ylabel('Dangerous Crimes per Square Mile')
plt.show()



#TOTAL CRIME PER CAPITA WITHOUT FINANCIAL DISTRIC
crime_distcat_nofin_sorted = crime_distcat_nofin.sort_values('Total Crime per_capita', ascending = True)
crime_distcat_nofin_sorted.loc['Presidio','Color'] = 'g'
crime_distcat_nofin_sorted.loc['Presidio Heights', 'Color'] = 'g'
crime_distcat_nofin_sorted.plot(y = 'Total Crime per_capita', kind = 'bar', color = crime_distcat_nofin_sorted['Color'])
plt.title('Total Crime Per Capita by Neighborhood (without Financial District)')
plt.ylabel('Number of Crimes Per Capita')
plt.show()

#TOTAL CRIME PER CAPITA
crime_distcat_sorted_pc = crime_distcat.sort_values('Total Crime per_capita', ascending = True)
crime_distcat_sorted_pc.loc['Presidio','Color'] = 'g'
crime_distcat_sorted_pc.loc['Presidio Heights', 'Color'] = 'g'
crime_distcat_sorted_pc.plot(y = 'Total Crime per_capita', kind = 'bar', color = crime_distcat_sorted_pc['Color'])
plt.title('Total Crime Per Capita by Neighborhood')
plt.ylabel('Number of Crimes Per Capita')
plt.show()


#TOTAL CRIME BY AREA
t_sorted_by_area = crime_distcat.sort_values('Total Crime per_area', ascending = True)
t_sorted_by_area.loc['Presidio', 'Color'] = 'g'
t_sorted_by_area.loc['Presidio Heights', 'Color'] = 'g'
t_sorted_by_area.plot(y = 'Total Crime per_area', kind = 'bar', color = t_sorted_by_area['Color'])
plt.title('Number of Total Crimes by Area')
plt.ylabel('Total Crimes per Square Mile')
plt.show()
