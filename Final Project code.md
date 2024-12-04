
# Final-Project-Code
1 IMPORTING LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from datetime import datetime
from prophet import Prophet

2 EXTRACTING THE DATA


gen1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv') # SOLAR GENARTION DATA
whet1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')# WHEATHER DATA
![image](https://github.com/user-attachments/assets/867566f5-e6bd-49d3-b40c-60bb98139b16)
3. LET'S SEE WHAT WE HAVE ON THE DATA SET


gen1.info()
gen1.tail(5)

print(gen1.isna().sum())
print(gen1.isnull().sum())
print(f'Unique number of inverters: {np.count_nonzero(gen1["SOURCE_KEY"].unique())}')
print(f'Unique number of plants: {np.count_nonzero(gen1["PLANT_ID"].unique())}')


print(whet1.info())
print(whet1.sample(5))


# DATE_TIME OBJECT CONVERTED TO datetime64 FOR USING IN TIME-SERIES ANALYSIS
gen1['DATE_TIME'] = pd.to_datetime(gen1['DATE_TIME'], dayfirst=True)
whet1['DATE_TIME'] = pd.to_datetime(whet1['DATE_TIME'], dayfirst=True)


gen1['DATE'] = pd.to_datetime(gen1['DATE_TIME'],format='%d-%m-%Y %H:%M').dt.date
gen1['TIME'] = pd.to_datetime(gen1['DATE_TIME'],format='%d-%m-%Y %H:%M').dt.time
gen1['HOUR'] = pd.to_datetime(gen1['DATE_TIME'],format='%d-%m-%Y %H:%M').dt.hour

gen1['DATE'] = dt.datetime



# WHEN MERGING DATA SETS SOME COLUMNS DUE TO HAVING 1 UNIQUE VALUE
merged_df = pd.merge(gen1.drop(columns='PLANT_ID'), whet1.drop(columns=['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

print(merged_df.info(), merged_df.sample(5))



grouped_df = merged_df.groupby(['TIME', 'SOURCE_KEY'])['DC_POWER'].mean().unstack()
source_key_averages = merged_df.groupby('SOURCE_KEY')['DC_POWER'].mean()
below_average_source_keys = source_key_averages[source_key_averages < merged_df['DC_POWER'].mean()].sort_values()
print(f'Under performing inverters:{below_average_source_keys}')


grouped_df.plot(figsize=(10,8))
plt.xlabel('TIME')
plt.ylabel('Average DC Power')
plt.title('Average DC Power Over Time for Each Source')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.show()
![image](https://github.com/user-attachments/assets/d2586fec-987c-48f6-8aa8-88c9cbab995d)



sns.lineplot(data=merged_df, x='DATE_TIME', y='DC_POWER')

plt.xlabel('Date')
plt.ylabel('Average DC Power')
plt.title('Average DC Power Over Time for Each Source')
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

 
plt.show()
![image](https://github.com/user-attachments/assets/f251a929-21d9-4ae0-b128-7066faaf82a4)



 
plt.show()
![image](https://github.com/user-attachments/assets/c1796cea-67da-44d7-9ac0-889899192931)

corr_matrix = merged_df.select_dtypes(include='number')

plt.figure(dpi=100, figsize=(8,8))
sns.heatmap(data=corr_matrix.corr(method='spearman'), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
![image](https://github.com/user-attachments/assets/1395d227-4322-4b0d-80f1-8b44924fda64)


