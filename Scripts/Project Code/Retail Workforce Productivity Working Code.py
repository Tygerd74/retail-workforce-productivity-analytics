import pandas as pd
import numpy as np


df = pd.read_csv("Retail student data.csv")

df.head()

df.info()

#no null values, all rows filled

df.describe()

#Check for item count consistency:
(df['nregitme'] + df['nbulkitem'] == df['nitem']).value_counts()



#Check for transaction time consistency:
df['calc_time'] = (
    pd.to_datetime(df['enddate'] + ' ' + df['endtime']) -
    pd.to_datetime(df['startdate'] + ' ' + df['starttime'])
).dt.total_seconds()

(df['calc_time'] == df['transtimeinsec']).value_counts()

#Making sure no negative or impossible values:
    
#(df[['nitem','total','transtimeinsec']] < 0).sum()
# came back with: '<' not supported between instances of 'str' and 'int',,, at least one of these is not stored as a numeric value
# see which one

df[['nitem','total','transtimeinsec']].dtypes
#total is a string

cols = ['nitem','total','transtimeinsec']

for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
df[cols].isnull().sum()
# now we have 1176 NaN values in "total"

(df[['nitem','total','transtimeinsec']] < 0).sum()
# issues with conversion, transtimeinsec is 0


#find potential issues in transaction time

df[df['transtimeinsec'] <= 1.5]
#several entries less than 1.5 seconds, most likely errors

#keep transactions over 1.5 seconds
df = df[df['transtimeinsec'] > 1.5]


#ACCOUNT FOR LONGER THAN 900 SECONDS

df[df['transtimeinsec'] >= 900]
#several entries less than 1.5 seconds, most likely errors

#keep transactions over 1.5 seconds
df = df[df['transtimeinsec'] < 900]




df[df['total'].isnull()][['total']].head(20)
df.loc[df['total'].isnull(), 'total_raw'] = df['total']

#Clean Total:
    
df['total'] = (
    df['total']
    .astype(str)
    .str.replace('$', '', regex=False)
    .str.replace(',', '', regex=False)
    .str.strip()
)

df['total'] = pd.to_numeric(df['total'], errors='coerce')

df['total'].isnull().sum()
#still 1176 issues, checking them more

total_issues = df[df['total'].isnull()]
#interesting, all of the total columns that were strings and are NaN now have a coupon used

df['total'].dtype
#decided to drop them:

df = df.dropna(subset=['total'])

df[cols].isnull().sum()
#now no null values again

#Check for some other stats:
    
#Make sure returns are correctly identified:
df[(df['total'] < 0) & (df['returndummy'] != 1)]
#4264 rows where total is positive despite return
check_df = df[(df['returndummy'] == 1) & (df['total'] >=0)].copy()

check_df[['nregitme', 'nbulkitem', 'nitem', 'total']].head(20)
#appear to have other items making the total a positive number, so makes sense


#Moving to check for outliers:
    
df['transtimeinsec'].describe()


#Create variable

df['datetime'] = pd.to_datetime(df['startdate'] + ' ' + df['starttime'])
df['hour_block'] = df['datetime'].dt.floor('H')
df['items_per_min'] = df['nitem'] / df['transtimeinsec'] * 60
df = df[df['items_per_min'] > 0].copy()
df['log_items_per_min'] = np.log(df['items_per_min'])

transactions_per_hour = (
    df.groupby('hour_block')
    .size()
    .rename('transactions_per_hour')
    .reset_index()
)

df = df.merge(transactions_per_hour, on='hour_block', how='left')


df['datetime'] = pd.to_datetime(
    df['startdate'].astype(str).str.strip() + ' ' + df['starttime'].astype(str).str.strip(),
    errors='coerce'
)

df = df.dropna(subset=['datetime']).copy()



items_per_hour = (
    df.groupby('hour_block')['nitem']
    .sum()
    .rename('items_per_hour')
    .reset_index()
)

df = df.merge(items_per_hour, on='hour_block', how='left')




#removes extreme outliers 

cols_to_winsorize = [
    'transtimeinsec',
    'items_per_min',
    'transactions_per_hour'
]

# add items_per_hour only if you've created it
if 'items_per_hour' in df.columns:
    cols_to_winsorize.append('items_per_hour')

for col in cols_to_winsorize:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=lower, upper=upper)
### ADDRESS ALL THE VARIABLES



# Checking categories of data
df['paymentmethod'].value_counts()
#4    352170
#1    313545
#3    289925
#6     24985
#9     11032
#2      7795
#8       347
#5       195
#7         6



#Checking for duplicate transactions
df.duplicated(subset=['invoice']).sum()
#576 duplicates in invoice

dupes = df[df.duplicated(subset=['invoice'], keep=False)]
dupes.sort_values('invoice')

dupes.head(20)

df['invoice'].value_counts().loc[lambda x: x > 1]

dupes.groupby('invoice').nunique()

#seems like most of the duplicates are for different transactions, not returns or anything related to each other
#going ahead and dropping

df = df.drop_duplicates(subset=['invoice'], keep='first')


#Ensure time-based 
    
df['datetime'] = pd.to_datetime(df['startdate'] + ' ' + df['starttime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek



#Create target variable (Checkout Speed) cleanly:
    
df['items_per_min'] = df['nitem'] / df['transtimeinsec'] * 60
df = df[df['items_per_min'] > 0].copy()
df['log_items_per_min'] = np.log(df['items_per_min'])





df['items_per_min'].sort_values(ascending=False).head(100)


#differentiate north and south storefronts
df['store_terminal'] = np.where(
    df['store_north'] == 1,
    "North_" + df['terminal'].astype(str),
    "South_" + df['terminal'].astype(str)
)




import matplotlib.pyplot as plt

df['items_per_min'].hist(bins=50)
plt.title("Items per Minute Distribution")
plt.xlim(0, 200)
plt.show()

df['log_items_per_min'].hist(bins=50)
plt.title("Log Items per Minute Distribution")
plt.show()







### REGRESSION

df['datetime'] = pd.to_datetime(
    df['startdate'].astype(str).str.strip() + ' ' +
    df['starttime'].astype(str).str.strip(),
    errors='coerce'
)

df = df.dropna(subset=['datetime']).copy()

df['date'] = df['datetime'].dt.floor('D')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek   # Monday=0
df['year'] = df['datetime'].dt.year
df['week'] = df['datetime'].dt.isocalendar().week.astype(int)
df['hour_block'] = df['datetime'].dt.floor('H')



# CLOPENING VARIABLE

df['is_closing'] = df['hour'].between(19, 22).astype(int)   # 7pm–10pm
df['is_opening'] = df['hour'].between(6, 8).astype(int)     # 6am–8am

# aggregate to cashier-day level
daily_flags = (
    df.groupby(['employeeid', 'date'])
    .agg(
        is_closing=('is_closing', 'max'),
        is_opening=('is_opening', 'max')
    )
    .reset_index()
)

daily_flags = daily_flags.sort_values(['employeeid', 'date'])
daily_flags['next_date'] = daily_flags['date'] + pd.Timedelta(days=1)

# merge next-day opening
next_day = daily_flags[['employeeid', 'date', 'is_opening']].rename(
    columns={'date': 'next_date', 'is_opening': 'next_opening'}
)

daily_flags = daily_flags.merge(
    next_day,
    on=['employeeid', 'next_date'],
    how='left'
)

# define clopening
daily_flags['clopening'] = (
    (daily_flags['is_closing'] == 1) &
    (daily_flags['next_opening'] == 1)
).astype(int)

# merge back to transaction level
df = df.merge(
    daily_flags[['employeeid', 'date', 'clopening']],
    on=['employeeid', 'date'],
    how='left'
)

df['clopening'] = df['clopening'].fillna(0)

# PREP TIME VARIABLES


df['datetime'] = pd.to_datetime(
    df['startdate'].astype(str).str.strip() + ' ' +
    df['starttime'].astype(str).str.strip(),
    errors='coerce'
)

df = df.dropna(subset=['datetime']).copy()

df['date'] = df['datetime'].dt.floor('D')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek   # Monday = 0
df['year'] = df['datetime'].dt.isocalendar().year.astype(int)
df['week'] = df['datetime'].dt.isocalendar().week.astype(int)

# create a sequential week id to make rolling windows easier
df['week_id'] = df['year'] * 100 + df['week']



# HOUR CONSISTENCY
# Consistency(hour) = DaysWorkedAtHour / TotalDaysWorked
# over prior 4 weeks


# one row per cashier-date-hour worked
cashier_date_hour = (
    df.groupby(['employeeid', 'date', 'hour'])
    .size()
    .reset_index(name='worked_hour')
)

cashier_date_hour['worked_hour'] = 1

# one row per cashier-date
cashier_date = (
    df.groupby(['employeeid', 'date'])
    .size()
    .reset_index(name='worked_day')
)

cashier_date['worked_day'] = 1

# attach week_id to both
date_to_week = (
    df[['employeeid', 'date', 'week_id']]
    .drop_duplicates()
)

cashier_date_hour = cashier_date_hour.merge(
    date_to_week,
    on=['employeeid', 'date'],
    how='left'
)

cashier_date = cashier_date.merge(
    date_to_week,
    on=['employeeid', 'date'],
    how='left'
)

# total days worked by cashier in each week
days_worked_week = (
    cashier_date.groupby(['employeeid', 'week_id'])['worked_day']
    .sum()
    .reset_index(name='days_worked_in_week')
)

# days worked at each hour by cashier in each week
days_worked_hour_week = (
    cashier_date_hour.groupby(['employeeid', 'week_id', 'hour'])['worked_hour']
    .sum()
    .reset_index(name='days_worked_at_hour_in_week')
)

# make full skeleton for employee-week-hour combinations
all_emp_week = days_worked_week[['employeeid', 'week_id']].drop_duplicates()
all_hours = pd.DataFrame({'hour': sorted(df['hour'].dropna().unique())})

all_emp_week['key'] = 1
all_hours['key'] = 1

hour_panel = all_emp_week.merge(all_hours, on='key').drop(columns='key')

hour_panel = hour_panel.merge(
    days_worked_week,
    on=['employeeid', 'week_id'],
    how='left'
)

hour_panel = hour_panel.merge(
    days_worked_hour_week,
    on=['employeeid', 'week_id', 'hour'],
    how='left'
)

hour_panel['days_worked_at_hour_in_week'] = hour_panel['days_worked_at_hour_in_week'].fillna(0)

hour_panel = hour_panel.sort_values(['employeeid', 'hour', 'week_id'])

# prior 4-week rolling sums
hour_panel['days_worked_at_hour_4w'] = (
    hour_panel.groupby(['employeeid', 'hour'])['days_worked_at_hour_in_week']
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).sum())
)

hour_panel['days_worked_total_4w'] = (
    hour_panel.groupby('employeeid')['days_worked_in_week']
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).sum())
)

hour_panel['consistency_hour'] = np.where(
    hour_panel['days_worked_total_4w'] > 0,
    hour_panel['days_worked_at_hour_4w'] / hour_panel['days_worked_total_4w'],
    np.nan
)

# merge hour consistency back to df using employeeid, week_id, hour
if 'consistency_hour' in df.columns:
    df = df.drop(columns=['consistency_hour'])

df = df.merge(
    hour_panel[['employeeid', 'week_id', 'hour', 'consistency_hour']],
    on=['employeeid', 'week_id', 'hour'],
    how='left'
)



# DAY CONSISTENCY
# Consistency(day) = WeeksWorkedOnDay / WeeksWorked
# over prior 4 weeks

# one row per cashier-week-day showing whether they worked that day
cashier_week_day = (
    df.groupby(['employeeid', 'week_id', 'day_of_week'])
    .size()
    .reset_index(name='worked_on_day')
)

cashier_week_day['worked_on_day'] = 1

# one row per cashier-week showing whether they worked at all that week
cashier_week = (
    df.groupby(['employeeid', 'week_id'])
    .size()
    .reset_index(name='worked_week')
)

cashier_week['worked_week'] = 1

# full skeleton for employee-week-day combinations
all_days = pd.DataFrame({'day_of_week': list(range(7))})

all_emp_week2 = cashier_week[['employeeid', 'week_id']].drop_duplicates().copy()
all_emp_week2['key'] = 1
all_days['key'] = 1

day_panel = all_emp_week2.merge(all_days, on='key').drop(columns='key')

day_panel = day_panel.merge(
    cashier_week,
    on=['employeeid', 'week_id'],
    how='left'
)

day_panel = day_panel.merge(
    cashier_week_day,
    on=['employeeid', 'week_id', 'day_of_week'],
    how='left'
)

day_panel['worked_on_day'] = day_panel['worked_on_day'].fillna(0)

day_panel = day_panel.sort_values(['employeeid', 'day_of_week', 'week_id'])

# prior 4-week rolling sums

#SWITCHING TO 2 FOR ROBUSTNESS CHECKS, normally a 4 after .rolling(
day_panel['weeks_worked_on_day_4w'] = (
    day_panel.groupby(['employeeid', 'day_of_week'])['worked_on_day']
    .transform(lambda x: x.shift(1).rolling(2, min_periods=1).sum())
)

day_panel['weeks_worked_total_4w'] = (
    day_panel.groupby('employeeid')['worked_week']
    .transform(lambda x: x.shift(1).rolling(2, min_periods=1).sum())
)

day_panel['consistency_day'] = np.where(
    day_panel['weeks_worked_total_4w'] > 0,
    day_panel['weeks_worked_on_day_4w'] / day_panel['weeks_worked_total_4w'],
    np.nan
)

# merge day consistency back to df using employeeid, week_id, day_of_week
if 'consistency_day' in df.columns:
    df = df.drop(columns=['consistency_day'])

df = df.merge(
    day_panel[['employeeid', 'week_id', 'day_of_week', 'consistency_day']],
    on=['employeeid', 'week_id', 'day_of_week'],
    how='left'
)

# quick checks
print(df[['consistency_hour', 'consistency_day']].describe())
print(df[['consistency_hour', 'consistency_day']].isnull().sum())




import statsmodels.formula.api as smf

df['employeeid'].nunique()

df_raw = df.dropna(subset =[
    'log_items_per_min',
    'transactions_per_hour',
    'nitem',
    'cashbackdummy',
    'coupondummy',
    'discdummy',
    'paymentmethod',
    'employeeid',
    'store_terminal'
]).copy()





df_hour = df.dropna(subset=[
    'log_items_per_min',
    'transactions_per_hour',
    'consistency_hour',
    'day_of_week',
    'nitem',
    'cashbackdummy',
    'coupondummy',
    'discdummy',
    'paymentmethod',
    'employeeid',
    'hour',
    'store_terminal'
]).copy()


df_day = df.dropna(subset=[
    'log_items_per_min',
    'transactions_per_hour',
    'consistency_day',
    'nitem',
    'cashbackdummy',
    'coupondummy',
    'discdummy',
    'paymentmethod',
    'employeeid',
    'day_of_week',
    'store_terminal'
]).copy()


df_mixed = df.dropna(subset=[
    'log_items_per_min',
    'transactions_per_hour',
    'consistency_day',
    'consistency_hour',
    'nitem',
    'cashbackdummy',
    'coupondummy',
    'discdummy',
    'paymentmethod',
    'employeeid',
    'day_of_week',
    'store_terminal'
]).copy()




model_raw = smf.ols(
    formula="""
    log_items_per_min ~
    C(nitem) +
    cashbackdummy +
    coupondummy +
    discdummy +
    C(paymentmethod) +
    clopening +
    C(store_terminal) +
    C(employeeid)
    """,
    data=df_raw
).fit(cov_type='HC3')

print("\n===== RAW MODEL (NOTHIN TO IT) =====")
print(model_raw.summary())
        


    
model_hour = smf.ols(
    formula="""
    log_items_per_min ~
    C(hour) +
    C(day_of_week) +
    consistency_hour +
    transactions_per_hour +
    C(nitem) +
    cashbackdummy +
    coupondummy +
    discdummy +
    C(paymentmethod) +
    clopening +
    C(store_terminal) +
    C(employeeid)
    """,
    data=df_hour
).fit(cov_type='HC3')

print("\n===== HOUR MODEL (NO CALENDAR DAY FE) =====")
print(model_hour.summary())


model_hour_dateFE = smf.ols(
    formula="""
    log_items_per_min ~
    C(hour) +
    C(date) +
    consistency_hour +
    transactions_per_hour +
    C(nitem) +
    cashbackdummy +
    coupondummy +
    discdummy +
    C(paymentmethod) +
    clopening +
    C(store_terminal) +
    C(employeeid)
    """,
    data=df_hour
).fit(cov_type='HC3')

print("\n===== HOUR MODEL (WITH CALENDAR DAY FE) =====")
print(model_hour_dateFE.summary())


#“I estimate two specifications: one capturing within-day variation using hour fixed effects, and another capturing weekly patterns using day-of-week fixed effects. Both models control for demand using transactions per hour, transaction complexity, and include cashier fixed effects. I additionally incorporate a clopening indicator to capture potential fatigue effects.”

model_day = smf.ols(
    formula="""
    log_items_per_min ~
    C(day_of_week) +
    consistency_day +
    transactions_per_hour +
    C(nitem) +
    cashbackdummy +
    coupondummy +
    discdummy +
    C(paymentmethod) +
    clopening +
    C(store_terminal) +
    C(employeeid)
    """,
    data=df_day
).fit(cov_type='HC3')

print("\n===== DAY-OF-WEEK MODEL =====")
print(model_day.summary())


#add interaction effect

model_hour_interaction = smf.ols(
    formula="""
    log_items_per_min ~
    consistency_hour * consistency_day +

    consistency_hour +

    consistency_day +

    C(hour)+
    C(day_of_week)+

    
    transactions_per_hour +

    
    C(nitem) +
    cashbackdummy +
    coupondummy +
    discdummy +
    C(paymentmethod) +

    
    clopening +

    
    C(store_terminal) +
    C(employeeid)
    """,
    data=df_mixed
).fit(cov_type='HC3')
#df_hour due to asymetric model 

print("\n===== HOUR × DAY INTERACTION MODEL =====")
print(model_hour_interaction.summary())


model_hour_day = smf.ols(
    formula="""
    log_items_per_min ~

    consistency_hour +

    consistency_day +

    C(hour)+
    C(day_of_week)+

    
    transactions_per_hour +

    
    C(nitem) +
    cashbackdummy +
    coupondummy +
    discdummy +
    C(paymentmethod) +

    
    clopening +

    
    C(store_terminal) +
    C(employeeid)
    """,
    data=df_hour
).fit(cov_type='HC3')
#df_hour due to asymetric model 
print("\n===== HOUR × DAY INTERACTION MODEL =====")
print(model_hour_day.summary())






### Measurable impact

df["consistency_hour"].mean()
df["consistency_day"].mean()


cashier_avg = df.groupby("employeeid")[["consistency_hour", "consistency_day"]].mean()

cashier_avg.mean()


#based off interaction effect model results:
    
# coefficients from your regression
beta_hotd = 0.0644
beta_dotw = 0.0570
beta_inter = -0.0622

# baseline (your actual dataset averages)
hotd_base = 0.440249
dotw_base = 0.603164

def predict_productivity(hotd, dotw):
    return (
        beta_hotd * hotd +
        beta_dotw * dotw +
        beta_inter * hotd * dotw
    )

# baseline prediction
y_base = predict_productivity(hotd_base, dotw_base)

# improved HOTD (+10 percentage points)
hotd_new = hotd_base +0.15
dotw_new = dotw_base + 0.10

y_new = predict_productivity(hotd_new, dotw_new)

# change
delta_y = y_new - y_base

# convert to percent
percent_change = delta_y * 100

print("Baseline Productivity (log):", round(y_base, 5))
print("New Productivity (log):", round(y_new, 5))
print("Change (log points):", round(delta_y, 5))
print("Percent Change (%):", round(percent_change, 3), "%")


# In terms of time savings:
    
avg_transaction_time_sec = 60  # adjust if needed

time_saved_per_tx = avg_transaction_time_sec * (percent_change / 100)

print("Time saved per transaction (seconds):", round(time_saved_per_tx, 4))

# Over a year:

# from previous step
percent_change = delta_y * 100

avg_transaction_time_sec = 60  # adjust if needed
time_saved_per_tx = avg_transaction_time_sec * (percent_change / 100)

transactions_per_day = 800   # per cashier (adjust if needed)
working_days_per_year = 250  # ~5 days/week * 50 weeks

time_saved_per_day = time_saved_per_tx * transactions_per_day
time_saved_per_year = time_saved_per_day * working_days_per_year

minutes_saved_per_year = time_saved_per_year / 60
hours_saved_per_year = minutes_saved_per_year / 60

print("Time saved per transaction (sec):", round(time_saved_per_tx, 4))
print("Time saved per day (sec):", round(time_saved_per_day, 2))
print("Time saved per year (hours):", round(hours_saved_per_year, 2))

# assess over total employees

num_cashiers = 126

total_store_hours_saved = hours_saved_per_year * num_cashiers

print("Total store-level hours saved per year:", round(total_store_hours_saved, 2))

