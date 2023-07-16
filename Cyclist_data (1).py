#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from pandas import Series, DataFrame


# In[2]:


import glob

glob.glob(r'/Users/harshitkoodi/Desktop/Cyclistic_Data/*.csv')


# In[3]:


all_dfs = []

for one_filename in glob.glob(r'/Users/harshitkoodi/Desktop/Cyclistic_Data/*.csv'):
    print(f'Loading {one_filename}')
    new_df = pd.read_csv(one_filename)
    
    all_dfs.append(new_df)


# In[4]:


df = pd.concat(all_dfs)


# In[5]:


df.tail(5)


# In[6]:


column_names = df.columns
print(column_names)


# In[7]:


df['member_casual'].value_counts()


# In[8]:


#sns.histplot(data=df, x='member_casual')


# In[9]:


from datetime import datetime

df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])


# In[10]:


df['ride_length'] = df['ended_at'] - df['started_at']


# In[11]:


df['ride_length']


# In[12]:


df['ride_length'].mean()


# In[13]:


df['day_of_week'] = df['started_at'].dt.dayofweek


# In[14]:


df['day_of_week'].head()
#monday 0 sunday 6


# In[15]:


null_count = df['member_casual'].isnull().sum()


# In[16]:


null_count


# In[17]:


mean_member = df.loc[df['member_casual'] == 'member', 'ride_length'].mean()
mean_casual = df.loc[df['member_casual'] == 'casual', 'ride_length'].mean()


# In[18]:


mean_casual


# In[19]:


mean_member


# In[20]:


from IPython.display import display, FileLink


# np.random.seed(42)
# 
# # Get the number of rows to drop
# num_rows_to_drop = int(len(df) * 0.2)
# 
# # Drop random rows
# df = df.drop(np.random.choice(df.index, num_rows_to_drop, replace=False))
# 
# # Save the modified DataFrame to a new file or overwrite the original file
# #df.to_csv('modified_file.csv', index=False)  # Replace 'modified_file.csv' with the desired file name

# df.to_csv('data.csv', index=False)
# 
# # Provide a download link for the CSV file
# display(FileLink('data.csv', result_html_prefix="Click here to download: "))

# In[21]:


df.shape


# In[22]:


riders_by_day = df.groupby(['day_of_week', 'member_casual']).size()


# In[23]:


riders_by_day


# In[37]:


import matplotlib.pyplot as plt

#riders_by_day = riders_by_day.unstack()
colors = {'casual' : 'orange', 'member' : 'green'}

# Plot the data

riders_by_day.unstack().plot(kind='bar', color = colors)

# Set the labels and title
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.title('Casual vs Regular Riders by Day of the Week')

# Show the plot
plt.show()


# In[25]:


from datetime import datetime

df['month'] = pd.to_datetime(df['started_at']).dt.month
    


# In[26]:


df['month']


# In[27]:


riders_by_month = df.groupby(['month', 'member_casual']).size()


# In[28]:


riders_by_month


# In[29]:


import matplotlib.pyplot as plt

# Plot the data
#riders_by_month.plot(kind='bar')

colors = {'casual': 'blue', 'member': 'red'}

# Plot the data

riders_by_month.unstack().plot(kind='bar', color=colors)

# Set the labels and title
plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Casual vs Member Riders by Month')

# Show the plot
plt.show()


# In[30]:


df['ride_length'] = pd.to_timedelta(df['ride_length'], unit='minutes')
riders_avgtime_byday = df.groupby(['day_of_week', 'member_casual'])['ride_length'].mean()
riders_avgtime_byday = riders_avgtime_byday.unstack()


# In[31]:


print(df['ride_length'].dtype)


# In[32]:


null_counts = riders_avgtime_byday.isnull().sum()
print(null_counts)


# #colors = {'casual': 'blue', 'member': 'red'}
# 
# # Plot the data
# riders_avgtime_byday.plot(kind='bar')
# 
# # Set the labels and title
# plt.xlabel('Day of Week')
# plt.ylabel('Average Ride Time')
# plt.title('Average Ride Time by Day of Week')
# 
# # Show the plot
# plt.show()

# In[39]:


bike_type = df.groupby(['rideable_type', 'member_casual']).size()


# In[40]:


bike_type


# In[41]:


colors = {'casual': 'blue', 'member': 'red'}

# Plot the data
bike_type.unstack().plot(kind='bar', color=colors)

# Set the labels and title
plt.xlabel('Bike Type')
plt.ylabel('Number of bikes')
plt.title('Which bike works the most')

# Show the plot
plt.show()


# In[ ]:




