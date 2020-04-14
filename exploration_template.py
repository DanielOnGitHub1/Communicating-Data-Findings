#!/usr/bin/env python
# coding: utf-8

# # Bay Wheels dataset (January, 2020) - Communicating Data Findings
# ## by Daniel Perez
# 
# ## Preliminary Wrangling
# 
# > Hello!<br>
# This exploratory effort will be made using a file containing numerous bike trips, which were made using Bay Wheels, a San Francisco bike share regional system owned by Lyft and operated by Motivate. The main purpose will be to analyze and uncover trends within the variables, enabling us to draw consistent conclusions and insights about the data, and to furthermore ellaborate a clean and comprehensible visual presentation concerning these findings. 

# ### Importing Libraries

# In[2]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the Dataset

# In[3]:


# Dataset already included in the notebook, extracted from Lyft's website:
bike_data = pd.read_csv('bay_wheels_data.csv')
bike_data.head()


# In[4]:


bike_data.info()


# ### As we can see, the dataset is composed by 295854 rows and 15 columns, concerning bike trips travel time info, its start and end station locations, bike id and user type, rental mode and even some Bike Share For All data, which is a campaign launched in early 2016 by San Francisco Bicycle Coalition, aimed at an equitable bike share system.
# 
# ### In this exploration, we'll address the 'where', the 'who', the 'when' and the 'how'. For that reason, these will be the variables of interest:
# #### - duration_sec (trip duration)
# #### - start_time
# #### - start_station_name
# #### - end_station_name
# #### - user_type
# #### - bike_share_for_all_trip
# #### - rental_access_method

# ## <font color='red'> Univariate Exploration<font>

# ### First things first, let's take a look at our variables of interest individually, to check their overall characteristics:

# ### <font color='blue'>Duration_min<font>

# ### Starting by the trip duration variable, let's pass it through a descriptive function:

# In[5]:


bike_data.duration_sec.describe()


# ### Before any valid observation, it is important to address that the maximum value is very distant from the third quartile value, possibly denouncing unrealistic occurrences. Let's take a closer look, and to allow for a more summarized result, let's transform the values from seconds to minutes:

# In[6]:


bike_data.duration_sec = (bike_data.duration_sec / 60).astype(int)


# ### Let's produce a histogram, ranging from 0 to 120 minutes, and count the total of occurrences:

# In[7]:


bike_data.duration_sec.hist(bins = [0, 15, 30, 45, 60, 75, 90, 105, 120]);


# In[8]:


bike_data.query('duration_sec <= 120').duration_sec.count()


# ### The histogram shows us the distribution, aligned with a count of 295159 values from 0 to 120 minutes, which indicates that almost all the values are located between 1 (the minimum value) to 120 minutes duration

# In[9]:


duration_120 = bike_data.query('duration_sec > 120').duration_sec.count()
duration_120/bike_data.shape[0] 


# ### In fact the vast majority of values is located between the 2-hour range, and exceeding values represent around 0,2 % of all elements, as displayed above. For that reason, let's remove the values that surpass the 120-minutes mark, in order to also remove any outliers that could mislead our conclusions:

# In[10]:


bike_data = bike_data[bike_data.duration_sec <= 120]


# In[11]:


bike_data.duration_sec.describe()


# ### With this we eliminated the odd ones and filtered the values allowing for a more realistic analysis. Let's produce another histogram with cleaner values:

# In[12]:


bike_data.duration_sec.hist(bins = [0, 10, 20 ,30, 40, 50, 60]);


# ### Indeed the distribution is centralized between 0 to 20 minutes, with a steep slope after it

# In[13]:


# renaming the column to 'duration_min':
bike_data.rename(columns={'duration_sec':'duration_min'}, inplace=True)


# ### <font color='blue'>Start_time<font>

# ### To better understand and work with this feature, let's split the datetime values into two separate columns (date and hour of the day):

# In[14]:


# converting the start_time column to datetime, and splitting into two columns

bike_data['day'] = bike_data['start_time'].str[8:10]
bike_data['hour_of_day'] = bike_data['start_time'].str[11:13]
bike_data['day'] = bike_data['day'].astype(int)
bike_data['hour_of_day'] = bike_data['hour_of_day'].astype(int)

# removing the 'start_time' column:

bike_data.drop(columns='start_time', inplace=True)

# rearranging the column order:

bike_data = bike_data[['duration_min', 
                       'day', 
                       'hour_of_day', 
                       'end_time', 
                       'start_station_id', 
                       'start_station_name',
                       'start_station_latitude',
                       'start_station_longitude',
                       'end_station_id',
                       'end_station_name',
                       'end_station_latitude',
                       'end_station_longitude',
                       'bike_id',
                       'user_type',
                       'bike_share_for_all_trip',
                       'rental_access_method']]


# In[15]:


bike_data.head()


# ### Good! now, let's explore these newly created columns, starting from the 'day' column:

# In[16]:


plt.figure(figsize=(12, 4))
sb.countplot(data = bike_data, x = 'day', color = 'cyan')
plt.title('Trip count per January day')
plt.xlabel('Day of January')
plt.ylabel('Trip Count');


# In[17]:


bike_data.day.value_counts().describe()


# ### Apparently there's a slight left skew, indicating that the number of trips rise to the end of the month. the mean of trips per day is 9521, and the median is 9193. Also, as displayed by the plot, there's a visible usage decrease on weekends (4-5, 11-12, 18-19 and 25-26)

# ### Now, to the 'hour_of_day' column:

# In[19]:


plt.figure(figsize=[10,4])
sb.countplot(data = bike_data, x = 'hour_of_day', color = 'orange')
plt.title('Trip Count Per Hour of Day')
plt.xlabel('Time of Day (hour)')
plt.ylabel('Trip Count');


# ### through this bar plot, we can observe a bimodal distribution, with peaks happening in the middle of the morning and also in the late afternoon, closer to beginning and end of work hours

# ### <font color='blue'>Start_station_name<font>

# In[112]:


bike_data.start_station_name.describe()


# ### We have 430 unique station names, with the most frequent station having a frequency of 2743 occurrences

# In[113]:


bike_data.start_station_name.isna().sum()/bike_data.shape[0]


# ### From all the rows, approximately half of 'start_station_name' values are missing

# ### <font color='blue'>End_station_name<font>

# In[114]:


bike_data.end_station_name.describe()


# ### We have 429 unique station names, with the most frequent station having a frequency of 5016 occurrences

# In[115]:


bike_data.end_station_name.isna().sum()/bike_data.shape[0]


# ### From all the rows, approximately half of 'end_station_name' values are missing

# ### <font color='blue'>User_type<font>

# ### Let's analyze the first dicotomical column:

# In[116]:


sb.countplot(data = bike_data, x = 'user_type', color = 'r');


# In[117]:


print('% of subscribers: {}'.format(bike_data.query('user_type == "Subscriber"').user_type.count()/bike_data.shape[0]))
print('% of customers: {}'.format(bike_data.query('user_type == "Customer"').user_type.count()/bike_data.shape[0]))


# ### As we can see, there's a bigger frequency of subscribers compared to casual users

# In[118]:


bike_data.user_type.isna().sum()


# ### No occurrence of missing values in this column!

# ### <font color='blue'>Bike_share_for_all<font>

# In[119]:


bike_data.bike_share_for_all_trip.unique()


# In[120]:


sb.countplot(data = bike_data, x= 'bike_share_for_all_trip');


# ### Through this bar chart, we can see a big differente between both values, as well as a lot of missing values. Let's take a look at their counts:

# In[121]:


bike_data.bike_share_for_all_trip.value_counts()


# In[122]:


bike_data.bike_share_for_all_trip.isna().sum()


# ### The 'yes' cases represent around 1% of valid values, and in parallel the majority of values are missing.

# ### <font color='blue'>Rental_access_method<font>

# In[123]:


bike_data.rental_access_method.unique()


# In[124]:


sb.countplot(data = bike_data, x= 'rental_access_method');


# ### According to this bar plot, the vast majority of values are represented by the app rental method. Let's zoom in on their actual values:

# In[125]:


bike_data.rental_access_method.value_counts()


# In[126]:


print('% of Clipper users: {}'.format(bike_data.rental_access_method.value_counts()[1]/
                                      bike_data.rental_access_method.value_counts()[0]))
print('% of App users: {}'.format(bike_data.rental_access_method.value_counts()[0]/
                                  bike_data.rental_access_method.value_counts().sum()))
                                     


# ### Indeed the clipper rental method represents 8% of trips, compared to 92% app uses

# ### To finalize our univariate exploration phase, let's drop the columns that we do not need:

# In[127]:


bike_data.drop(columns=['end_time',
                        'start_station_id',
                        'start_station_latitude',
                        'start_station_longitude',
                        'end_station_id',
                        'end_station_latitude',
                        'end_station_longitude',
                        'bike_id'], inplace=True)


# In[128]:


bike_data.columns


# ### All set!

# ## <font color='red'>Bivariate Exploration<font>

# ### As our first bivariate analysis, let's compare the relationship between the duration and time of day, considering trip occurrences:

# In[129]:


plt.hist2d(data = bike_data, x = 'day', y = 'hour_of_day', cmap = 'viridis_r')
plt.colorbar()
plt.title('Time of day vs January day - Trip count')
plt.xlabel('January Day')
plt.ylabel('Time of Day (hour)')
plt.yticks(np.arange(0,25,4));


# ### As we can see by this heatmap, the already observed consistency in the start time on the mid-morning (close to 8 am) and mid to late afternoon (close to 17 pm) endure during the whole month, accentuating during time. The latter can be related to the trip count increase towards the end of the month, represented by the ascending count pattern seen in the trip count by day plot on the univariate exploration.

# ### What about the members and non-members of the Bike Share For All program? Do the start time hours follow the same pattern from above based on this feature?

# In[130]:


x_bins = np.arange(0, 25, 2)
sb.violinplot(data = bike_data, y = 'bike_share_for_all_trip', x = 'hour_of_day', palette='Spectral')
plt.xticks(x_bins);


# ### As we can see from this violin plot, the members present a different behaviour concerning their bike use, showing a more homogeneous characteristic compared to the non-members bimodal pattern, peaking at around 4 pm

# ### furthering down the hour_of_day investigation, let's observe the habits of both user types concerning the trip start hour, comparing their frequencies side by side:

# In[131]:


plt.figure(figsize=(12,4))
sb.countplot(data = bike_data, x = 'hour_of_day', hue = 'user_type')
plt.title('Start time trip count per User Type')
plt.xlabel('Time of Day (hour)')
plt.ylabel('Trip Count');


# ### This clustered bar chart unravels a visible proportionality between them throughout the day, both following the same bimodal pattern of mid-morning and mid-to-end afternoon, with the subscribers always in bigger numbers than the casual customers.

# ### Jumping to another kind of feature pair, let's check the relation between the 2 types of users and the 2 types of access methods:

# In[132]:


sb.countplot(data = bike_data, x = 'rental_access_method', hue = 'user_type', palette='magma')
plt.title('Trip count per User Type and Rental Method')
plt.xlabel('Rental Method')
plt.ylabel('Trip Count')
plt.legend(title='User Type');
bike_data.rental_access_method.value_counts()


# ### As described by the plot and value counts table, the app method is the preferred choice, however there's still a significant amount of clipper method subscribers.

# ### Now, let's check the trip duration of both user types:

# In[138]:


plt.figure(figsize=(12,4))
sb.violinplot(data = bike_data, y = 'user_type', x = 'duration_min', palette='GnBu')
x_ticks = np.arange(0, 120, 5)
plt.xticks(x_ticks)
plt.title('Trip duration mean per User Type')
plt.xlabel('Trip Duration - Mean')
plt.ylabel('User Type');
bike_data.duration_min.describe()


# ### Through the plot and the descriptive results, it is clear that both user types have very similar trip duration patterns, with the biggest concentration happening at around 5 to 7 minutes and the mean being around 12 minutes. <br><br>It is worth mentioning the presence of outliers that may mislead the mean, seeing that the 3rd quartile value is 15, but the maximum extends as further as 120. Nevertheless, having that the median is 9 minutes (2nd quartile), the mean is not so distant from a plausible representation.

# ### Continuing our exploration, let's discover which stations are used the most:

# In[89]:


### Plotting the 5 most used start stations
plt.subplot(2, 1, 1)
top_5_start = bike_data.start_station_name.value_counts()[:5]
top_5_start.plot(kind='barh')
plt.title('Top 5 Start Stations - Trip count')
plt.xlabel('Trip Count')

plt.xticks()

### Plotting the 5 most used end stations
plt.subplot(2, 1, 2)
plt.subplots_adjust(bottom = -0.5, hspace = 0.4)
top_5_end = bike_data.end_station_name.value_counts()[:5]
top_5_end.plot(kind='barh', color='orange')
plt.title('Top 5 End Stations - Trip count')
plt.xticks()
plt.xlabel('Trip Count');


# ### Interesting...with the exception of Howard St at Beale St station, all the other top 5 most used start stations repeat themselves in these results, with the addition of the Montgomery St BART Station. It is also notable the significant predominancy of the San Francisco Caltrain station trip count

# In[42]:


### There is a significant trip count among the top 5 most used stations. Let's see their proportion compared to all trips
print('% of trips starting on the top 5 most used start stations: {}'
      .format((bike_data.start_station_name.value_counts()[:5].sum()/bike_data.shape[0]*100)))
print('% of trips starting on the top 5 most used end stations: {}'
      .format((bike_data.end_station_name.value_counts()[:5].sum()/bike_data.shape[0]*100)))


# ### Approximately 4% of all trips start from the top 5 most used bike stations, all with quite similar trip counts. About the end stations, the top 5 represent approximately 5,5%

# ## <font color='red'>Multivariable Exploration<font>

# ### To start, let's plot a pair grid to spot any patterns through a macro perspective:

# In[43]:


g = sb.PairGrid(data = bike_data, x_vars = ['duration_min', 'day', 'hour_of_day'],
                y_vars = ['user_type','bike_share_for_all_trip', 'rental_access_method'])
plt.subplots_adjust(left=-0.5)
g.map(sb.violinplot, inner = 'quartile');


# ### In this pairplot is displayed the relations between the categorical and numerical values, with some already investigated in the bivariate exploration. These were the new revealed bivariate relations:
# ### <font color='red'>- Bike Share For All trips with day of trip shows a more stable consistency between members of the program<font>
# ### <font color='red'>- User type with day of trip shows an use increase by the subscribers along the month<font>
# ### <font color='red'>- Rental access method with day of trip also displays an use increase by both app and clipper users during the month<font><br><br>

# ### Furthering down our analysis, we can deepen our understanting about the data by taking more variables into consideration.

# ### Let's take a closer look at the Bike Share For All variable, taking into consideration the subscribers increase during the month:

# In[85]:


subs_by_day = bike_data.query('user_type == "Subscriber"').groupby('day').day.count()
bike_for_all_by_day = bike_data.query('bike_share_for_all_trip == "Yes"').groupby('day').day.count()

plt.subplot(1, 2, 1)
plt.plot(subs_by_day)
plt.xlabel('Day of January')
plt.ylabel('Trip Count')
y_ticks_1 = np.arange(0,10001,1000)
plt.yticks(y_ticks_1)
plt.title('Trip count - Subscribers')

plt.subplot(1, 2, 2)
plt.subplots_adjust(left=-1)
plt.plot(bike_for_all_by_day, color='r')
plt.xlabel('Day of January')
y_ticks_2 = np.arange(0,201,20)
plt.yticks(y_ticks_2)
plt.title('Trip count - Bike Share For All');


# ### <font color='red'>The subscribers growth tendency doesn't seem to push the Bike Share For All trip counts.  In fact, the Bike Share For All peak in the first days was the period of the smallest subscriber counts, which later on reached frequencies around 10000 in the last days, 5 times more than the beginning of the month<font>

# ### Taking into consideration the kind of user, the day and duration of trip, let's dive deeper:

# In[45]:


plt.figure(figsize=[10,4]);
sb.pointplot(data = bike_data, x = 'day', y = 'duration_min', hue = 'user_type')
plt.legend(bbox_to_anchor=[1.2,0.7], title='User Type')
plt.ylim(0,20)
plt.yticks(np.arange(0,20,2));


# ### <font color='red'>- During the month, the Customer user type presented bigger trip duration means peaks, overall always staying at least equal to the Subscribers means <font>

# In[46]:


plt.figure(figsize=[10,4]);
sb.pointplot(data = bike_data, x = 'hour_of_day', y = 'duration_min', hue = 'user_type')
plt.legend(bbox_to_anchor=[1.2,0.7], title='User Type')
plt.ylim(0,20)
plt.yticks(np.arange(0,20,2));


# ### <font color='red'>- There is certain homogeneity concerning the trip duration mean during the day for both user types, despite a one-week customer elevation between day 10 and 17<font><br>

# ## <font color='red'>Conclusions<font>

#    After passing through the univariable, bivariable and multivariable analyses, we are now able to infer certain characteristics and tendencies about the features, alongside with the Bay Wheels evolution through the month of January:
# - The use frequency increased throughout the month, approximately 5x from the first to the last day of the month, presenting superior counts on workdays
# - The start trip hour peaks at 8 am and 5 pm, suggesting a major use habit of going and returning from work.
# - Trip durations peak between 5 to 7 minutes. Taking the IQR into consideration, they have an overall 6 to 15 minutes duration range. The main focus is reasonably short trips.
# - Bike Share For All program have not presented an ascending adherence during the month, despite the rising subscribers numbers, which in fact represent the vasy majority of users. In parallel, the program members make trips more consistingly across all the day hours.
# - Both rental methods increased homogeneously, and besides the App having much more representativeness, there's still a considerable number of Clipper users.<br><br>
#     In summary, based on the January trips data, Bay Wheels showed a consistent overall growth, aligned with rising subscribers numbers, with a big applicability as work route transportation but with thousands of users during the weekend aswell.  Due to its large numbers and similar use increase, both app and clipper card methods need to be maintained to fulfill different needs and user profiles. The Bike Share For All program, with numbers representing around 1,5% of all trips, could be revised for a larger user adherence, aiming for an even more substantial evolution of the bike share option across San Francisco. Dissolving into roughly 10000 trips a day, these impressing statistics can represent the population's escalating search for a more affordable, eco-friendly and healthy way of transportation.

# In[ ]:




