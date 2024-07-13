import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


data = [
    {'season': 2023, 'player': 'erling haaland', 'team': 'manchester city', 'goals': 36},
    {'season': 2023, 'player': 'harry kane', 'team': 'tottenham hotspur', 'goals': 30},
    {'season': 2023, 'player': 'ivan toney', 'team': 'brentford', 'goals': 20},
    {'season': 2023, 'player': 'marcus rashford', 'team': 'manchester united', 'goals': 17},
    {'season': 2023, 'player': 'mohamed salah', 'team': 'liverpool', 'goals': 16},
    {'season': 2022, 'player': 'mohamed salah', 'team': 'liverpool', 'goals': 23},
    {'season': 2022, 'player': 'son heung-min', 'team': 'tottenham hotspur', 'goals': 23},
    {'season': 2022, 'player': 'cristiano ronaldo', 'team': 'manchester united', 'goals': 18},
    {'season': 2022, 'player': 'harry kane', 'team': 'tottenham hotspur', 'goals': 17},
    {'season': 2022, 'player': 'sadio mane', 'team': 'liverpool', 'goals': 16},
    {'season': 2021, 'player': 'harry kane', 'team': 'tottenham hotspur', 'goals': 23},
    {'season': 2021, 'player': 'mohamed salah', 'team': 'liverpool', 'goals': 22},
    {'season': 2021, 'player': 'bruno fernandes', 'team': 'manchester united', 'goals': 18},
    {'season': 2021, 'player': 'patrick bamford', 'team': 'leeds united', 'goals': 17},
    {'season': 2021, 'player': 'son heung-min', 'team': 'tottenham hotspur', 'goals': 17},
    {'season': 2020, 'player': 'jamie vardy', 'team': 'leicester city', 'goals': 23},
    {'season': 2020, 'player': 'pierre-emerick aubameyang', 'team': 'arsenal', 'goals': 22},
    {'season': 2020, 'player': 'danny ings', 'team': 'southampton', 'goals': 22},
    {'season': 2020, 'player': 'raheem sterling', 'team': 'manchester city', 'goals': 20},
    {'season': 2020, 'player': 'mohamed salah', 'team': 'liverpool', 'goals': 19},
    {'season': 2019, 'player': 'pierre-emerick aubameyang', 'team': 'arsenal', 'goals': 22},
    {'season': 2019, 'player': 'sadio mane', 'team': 'liverpool', 'goals': 22},
    {'season': 2019, 'player': 'mohamed salah', 'team': 'liverpool', 'goals': 22},
    {'season': 2019, 'player': 'sergio aguero', 'team': 'manchester city', 'goals': 21},
    {'season': 2019, 'player': 'jamie vardy', 'team': 'leicester city', 'goals': 18},
    {'season': 2018, 'player': 'mohamed salah', 'team': 'liverpool', 'goals': 32},
    {'season': 2018, 'player': 'harry kane', 'team': 'tottenham hotspur', 'goals': 30},
    {'season': 2018, 'player': 'sergio aguero', 'team': 'manchester city', 'goals': 21},
    {'season': 2018, 'player': 'raheem sterling', 'team': 'manchester city', 'goals': 20},
    {'season': 2018, 'player': 'jamie vardy', 'team': 'leicester city', 'goals': 18},
    {'season': 2017, 'player': 'harry kane', 'team': 'tottenham hotspur', 'goals': 29},
    {'season': 2017, 'player': 'romelu lukaku', 'team': 'everton', 'goals': 25},
    {'season': 2017, 'player': 'alexis sanchez', 'team': 'arsenal', 'goals': 24},
    {'season': 2017, 'player': 'sergio aguero', 'team': 'manchester city', 'goals': 20},
    {'season': 2017, 'player': 'diego costa', 'team': 'chelsea', 'goals': 20},
    {'season': 2016, 'player': 'harry kane', 'team': 'tottenham hotspur', 'goals': 25},
    {'season': 2016, 'player': 'sergio aguero', 'team': 'manchester city', 'goals': 24},
    {'season': 2016, 'player': 'jamie vardy', 'team': 'leicester city', 'goals': 24},
    {'season': 2016, 'player': 'romelu lukaku', 'team': 'everton', 'goals': 18},
    {'season': 2016, 'player': 'riyad mahrez', 'team': 'leicester city', 'goals': 17},
    {'season': 2015, 'player': 'sergio aguero', 'team': 'manchester city', 'goals': 26},
    {'season': 2015, 'player': 'harry kane', 'team': 'tottenham hotspur', 'goals': 21},
    {'season': 2015, 'player': 'diego costa', 'team': 'chelsea', 'goals': 20},
    {'season': 2015, 'player': 'charlie austin', 'team': 'queens park rangers', 'goals': 18},
    {'season': 2015, 'player': 'alexis sanchez', 'team': 'arsenal', 'goals': 16},
    {'season': 2014, 'player': 'luis suarez', 'team': 'liverpool', 'goals': 31},
    {'season': 2014, 'player': 'daniel sturridge', 'team': 'liverpool', 'goals': 21},
    {'season': 2014, 'player': 'yaya toure', 'team': 'manchester city', 'goals': 20},
    {'season': 2014, 'player': 'wayne rooney', 'team': 'manchester united', 'goals': 17},
    {'season': 2014, 'player': 'olivier giroud', 'team': 'arsenal', 'goals': 16},
    {'season': 2013, 'player': 'robin van persie', 'team': 'manchester united', 'goals': 26},
    {'season': 2013, 'player': 'luis suarez', 'team': 'liverpool', 'goals': 23},
    {'season': 2013, 'player': 'gareth bale', 'team': 'tottenham hotspur', 'goals': 21},
    {'season': 2013, 'player': 'christian benteke', 'team': 'aston villa', 'goals': 19},
    {'season': 2013, 'player': 'michu', 'team': 'swansea city', 'goals': 18},
    {'season': 2012, 'player': 'robin van persie', 'team': 'arsenal', 'goals': 30},
    {'season': 2012, 'player': 'wayne rooney', 'team': 'manchester united', 'goals': 27},
    {'season': 2012, 'player': 'sergio aguero', 'team': 'manchester city', 'goals': 23},
    {'season': 2012, 'player': 'emmanuel adebayor', 'team': 'tottenham hotspur', 'goals': 17},
    {'season': 2012, 'player': 'clint dempsey', 'team': 'fulham', 'goals': 17},
    {'season': 2011, 'player': 'dimitar berbatov', 'team': 'manchester united', 'goals': 20},
    {'season': 2011, 'player': 'carlos tevez', 'team': 'manchester city', 'goals': 20},
    {'season': 2011, 'player': 'darren bent', 'team': 'aston villa', 'goals': 17},
    {'season': 2011, 'player': 'robin van persie', 'team': 'arsenal', 'goals': 18},
    {'season': 2011, 'player': 'peter odemwingie', 'team': 'west bromwich albion', 'goals': 15},
    {'season': 2010, 'player': 'didier drogba', 'team': 'chelsea', 'goals': 29},
    {'season': 2010, 'player': 'wayne rooney', 'team': 'manchester united', 'goals': 26},
    {'season': 2010, 'player': 'darren bent', 'team': 'sunderland', 'goals': 24},
    {'season': 2010, 'player': 'carlos tevez', 'team': 'manchester city', 'goals': 23},
    {'season': 2010, 'player': 'frank lampard', 'team': 'chelsea', 'goals': 22},
    {'season': 2009, 'player': 'nicolas anelka', 'team': 'chelsea', 'goals': 19},
    {'season': 2009, 'player': 'cristiano ronaldo', 'team': 'manchester united', 'goals': 18},
    {'season': 2009, 'player': 'steven gerrard', 'team': 'liverpool', 'goals': 16},
    {'season': 2009, 'player': 'fernando torres', 'team': 'liverpool', 'goals': 14},
    {'season': 2009, 'player': 'robinho', 'team': 'manchester city', 'goals': 14},
    {'season': 2008, 'player': 'cristiano ronaldo', 'team': 'manchester united', 'goals': 31},
    {'season': 2008, 'player': 'emmanuel adebayor', 'team': 'arsenal', 'goals': 24},
    {'season': 2008, 'player': 'fernando torres', 'team': 'liverpool', 'goals': 24},
    {'season': 2008, 'player': 'robbie keane', 'team': 'tottenham hotspur', 'goals': 15},
    {'season': 2008, 'player': 'benjani', 'team': 'portsmouth', 'goals': 15},
    {'season': 2007, 'player': 'didier drogba', 'team': 'chelsea', 'goals': 20},
    {'season': 2007, 'player': 'benni mccarthy', 'team': 'blackburn rovers', 'goals': 18},
    {'season': 2007, 'player': 'cristiano ronaldo', 'team': 'manchester united', 'goals': 17},
    {'season': 2007, 'player': 'frank lampard', 'team': 'chelsea', 'goals': 11},
    {'season': 2007, 'player': 'wayne rooney', 'team': 'manchester united', 'goals': 14},
    {'season': 2006, 'player': 'thierry henry', 'team': 'arsenal', 'goals': 27},
    {'season': 2006, 'player': 'ruud van nistelrooy', 'team': 'manchester united', 'goals': 21},
    {'season': 2006, 'player': 'robbie keane', 'team': 'tottenham hotspur', 'goals': 16},
    {'season': 2006, 'player': 'darren bent', 'team': 'charlton athletic', 'goals': 18},
    {'season': 2006, 'player': 'frank lampard', 'team': 'chelsea', 'goals': 16},
    {'season': 2005, 'player': 'thierry henry', 'team': 'arsenal', 'goals': 25},
    {'season': 2005, 'player': 'andy johnson', 'team': 'crystal palace', 'goals': 21},
    {'season': 2005, 'player': 'robbie keane', 'team': 'tottenham hotspur', 'goals': 17},
    {'season': 2005, 'player': 'jermain defoe', 'team': 'tottenham hotspur', 'goals': 13},
    {'season': 2005, 'player': 'frank lampard', 'team': 'chelsea', 'goals': 13}
]


# Convert data to a DataFrame
df = pd.DataFrame(data)

# Confirm successful data loading
print(df.head())

# Data Analysis and Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set drawing style
sns.set(style="whitegrid")

# Calculate the average number of goals scored per season
avg_goals_per_season = df.groupby('season')['goals'].mean().reset_index()

# Draw a trend chart of average goals scored
plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_goals_per_season, x='season', y='goals', label='Overall Average')
plt.title('Average Goals per Season')
plt.xlabel('Season')
plt.ylabel('Average Goals')
plt.legend()
plt.show()

# Draw a box plot to display the distribution of goals
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='season', y='goals')
plt.title('Distribution of Goals per Season')
plt.xlabel('Season')
plt.ylabel('Goals')
plt.show()

pivot_table = df.pivot_table(index='player', columns='season', values='goals', aggfunc='sum')

# Draw a heat map to display player goals
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt="g")
plt.title('Goals by Season and Player')
plt.show()

# Construction and Evaluation of Predictive Models
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Select a player's data for a predictive model
player_data = df[df['player'] == 'harry kane']
player_data = player_data.set_index('season')['goals']

# Split the training and testing sets
train_size = int(len(player_data) * 0.8)
train, test = player_data[:train_size], player_data[train_size:]

# Constructing and fitting ARIMA models
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()  # Remove the 'disp' argument

# Making Predictions
forecast = model_fit.forecast(steps=len(test))


error = mean_squared_error(test, forecast)
print(f'Mean Squared Error: {error}')

# Predicting the Next 10 Years
future_forecast = model_fit.forecast(steps=10)

# Combining historical data with predictive data
future_years = [player_data.index[-1] + i + 1 for i in range(10)]
future_df = pd.DataFrame({'season': future_years, 'goals': future_forecast})

# Draw prediction results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, forecast, label='Forecast')
plt.plot(future_df['season'], future_df['goals'], label='Future Forecast', linestyle='--')
plt.title('Goals Prediction using ARIMA')
plt.xlabel('Season')
plt.ylabel('Goals')
plt.legend()
plt.show()