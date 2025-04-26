import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from io import BytesIO
import streamlit as st
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Formula1 vs Team's Stock Price Analysis",
    layout="wide",
    initial_sidebar_state="expanded"  
    )
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
plt.style.use('ggplot')
sns.set_style('whitegrid')
st.markdown("""
    <style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: white;
        background-color: #4B5D78;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-container {
        background-color: #D6DCE5;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 16px;
        margin-top: 10px;
    }
    .stExpander {
        border: none !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


F1_TEAMS = {
    "Ferrari": "RACE",
    "Mercedes": "MBGAF",
    "Aston Martin": "AML.L",
    "Red Bull": "ORCL",      
}
constructor_mapping = {
    'Ferrari': 'Ferrari',
    'Mercedes': 'Mercedes',
    'McLaren': 'McLaren',
    'Red Bull': 'Red Bull',
    'Alpine F1 Team': 'Alpine',
    'Aston Martin': 'Aston Martin',
    'Williams': 'Williams',
    'Alfa Romeo': 'Alfa Romeo',
    'AlphaTauri': 'AlphaTauri',
    'Haas F1 Team': 'Haas',
    'Racing Bulls': 'VCARB'
}

# Time periods for analysis
START_YEAR = 2020
print(f"Starting analysis for {START_YEAR}...")
END_YEAR = 2024
ANALYSIS_PERIOD = f"{START_YEAR}-{END_YEAR}"


data_path = "../Data"  # Path to the data directory
circuits = pd.read_csv(f"{data_path}/circuits.csv")
constructor_results = pd.read_csv(f"{data_path}/constructor_results.csv")
constructor_standings = pd.read_csv(f"{data_path}/constructor_standings.csv")
constructors = pd.read_csv(f"{data_path}/constructors.csv")
drivers = pd.read_csv(f"{data_path}/drivers.csv")
races = pd.read_csv(f"{data_path}/races.csv")
results = pd.read_csv(f"{data_path}/results.csv")

print(f"Loaded data successfully:")
print(f"- {len(races)} races from {races['year'].min()} to {races['year'].max()}")
print(f"- {len(constructors)} constructors")
print(f"- {len(drivers)} drivers")

f1_data =  {
    'circuits': circuits,
    'constructor_results': constructor_results,
    'constructor_standings': constructor_standings,
    'constructors': constructors,
    'drivers': drivers,
    'races': races,
    'results': results
}



def fetch_f1_race_results(year, f1_data):
    races_df = f1_data['races']
    results_df = f1_data['results']
    constructors_df = f1_data['constructors']
    
    year_races = races_df[races_df['year'] == year]
    
    if year_races.empty:
        print(f"No race data found for {year}")
        return pd.DataFrame()
    
    # Get race IDs for the specified year
    race_ids = year_races['raceId'].tolist()
    
    # Filter results for these races
    year_results = results_df[results_df['raceId'].isin(race_ids)]
    
    # Merge with races to get race names and dates
    results_with_races = pd.merge(
        year_results,
        year_races[['raceId', 'name', 'date', 'year', 'round']],
        on='raceId'
    )
    
    # Merge with constructors to get team names
    results_with_teams = pd.merge(
        results_with_races,
        constructors_df[['constructorId', 'name']],
        left_on='constructorId',
        right_on='constructorId'
    )
    
    results_with_teams = results_with_teams.rename(columns={
        'name_x': 'race_name',
        'name_y': 'team',
        'grid': 'qualifying_position',
        'position': 'race_position',
        'date': 'race_date'
    })
    
    # Convert race_position to numeric 
    results_with_teams['race_position'] = pd.to_numeric(
        results_with_teams['race_position'], 
        errors='coerce'
    )
    
    # Determine fastest lap
    if 'fastestLap' in results_with_teams.columns:
        results_with_teams['fastest_lap'] = results_with_teams['fastestLap'] == "1"
    else:
        # If no fastestLap column, we'll create a placeholder
        results_with_teams['fastest_lap'] = False
    
    final_results = results_with_teams[[
         'race_name', 'race_date', 'team', 
        'qualifying_position', 'race_position', 'points', 'fastest_lap'
    ]]
    
    # Convert race_date to datetime
    final_results['race_date'] = pd.to_datetime(final_results['race_date'])
    
    final_results['season'] = year
    
    print(f"Extracted {len(final_results)} results for the {year} F1 season")
    
    return final_results

def fetch_stock_data(tickers, start_date, end_date):
    
    all_data = pd.DataFrame()
    
    # Filter out None values (private companies)
    valid_tickers = [t for t in tickers if t]
    
    if not valid_tickers:
        return all_data
    
    # Download all stock data at once
    data = yf.download(valid_tickers, start=start_date, end=end_date,auto_adjust=False)
    
    # Process the data
    for ticker in valid_tickers:
        if len(valid_tickers) == 1:
            ticker_data = data.copy()
        else:
            ticker_data = data.xs(ticker, level=1, axis=1).copy()
        
        ticker_data['symbol'] = ticker
        ticker_data['date'] = ticker_data.index
        
        # Calculate returns
        ticker_data['daily_return'] = ticker_data['Adj Close'].pct_change()
        ticker_data['weekly_return'] = ticker_data['Adj Close'].pct_change(5)
        ticker_data['monthly_return'] = ticker_data['Adj Close'].pct_change(21)
        
        all_data = pd.concat([all_data, ticker_data.reset_index(drop=True)])
    
    # Map tickers to team names
    ticker_to_team = {v: k for k, v in F1_TEAMS.items() if v is not None}
    all_data['team'] = all_data['symbol'].map(ticker_to_team)
    
    return all_data

def fetch_f1_news(team, start_date, end_date, limit=50):
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if team == 'Mercedes':
        avg_articles_per_day = 2.5
        base_sentiment = 0.6
    elif team == 'Ferrari':
        avg_articles_per_day = 3.0
        base_sentiment = 0.55
    elif team == 'Red Bull':
        avg_articles_per_day = 2.8
        base_sentiment = 0.65
    else:
        avg_articles_per_day = 1.2
        base_sentiment = 0.5
    
    
    news_data = []
    for date in date_range:
        # News volume varies by team popularity and random factors
        n_articles = max(0, int(np.random.poisson(avg_articles_per_day)))
        
        # Skip some days with no news
        if n_articles == 0 and np.random.random() < 0.3:
            continue
            
        for i in range(n_articles):
            sentiment = min(1.0, max(-1.0, base_sentiment + np.random.normal(0, 0.3)))
            
            if sentiment > 0.7:
                headline_template = f"Positive news for {team}: {{event}}"
                events = [
                    "Promising test results", 
                    "New sponsorship deal announced",
                    "Technical breakthrough discovered",
                    "Driver praised for recent performance",
                    "Management restructuring brings optimism"
                ]
            elif sentiment > 0.3:
                headline_template = f"{team} {{event}}"
                events = [
                    "preparing well for upcoming race",
                    "meets expectations in practice session",
                    "maintains development pace",
                    "confident about future results",
                    "working on gradual improvements"
                ]
            elif sentiment > -0.3:
                headline_template = f"{team} {{event}}"
                events = [
                    "faces mixed results in testing",
                    "shows inconsistent performance",
                    "dealing with minor setbacks",
                    "working through technical challenges",
                    "reports no major changes"
                ]
            else:
                headline_template = f"Challenges for {team}: {{event}}"
                events = [
                    "Technical issues delay development",
                    "Poor weekend performance raises concerns",
                    "Internal conflicts reported",
                    "Sponsorship uncertainty emerges",
                    "Regulatory investigation underway"
                ]
            
            headline = headline_template.format(event=np.random.choice(events))
            
            sources = ["Autosport", "Motorsport", "Formula1.com", "RaceFans", "The Race"]
            
            news_data.append({
                'date': date,
                'team': team,
                'headline': headline,
                'sentiment': sentiment,
                'source': np.random.choice(sources)
            })
    
    return pd.DataFrame(news_data)

def collect_all_data(START_YEAR, END_YEAR, f1_data, F1_TEAMS):
    
    # Set date range
    start_date = f"{START_YEAR}-01-01"
    end_date = f"{END_YEAR}-12-31"
    
    # Collect race results
    print(f"Collecting race results for {START_YEAR}-{END_YEAR}...")
    race_results = pd.DataFrame()
    for year in range(START_YEAR, END_YEAR + 1):
        year_results = fetch_f1_race_results(year, f1_data)
        race_results = pd.concat([race_results, year_results])
    
    # Collect stock data
    print("Collecting stock data...")
    stock_data = fetch_stock_data(list(set(F1_TEAMS.values())), start_date, end_date)
    
    # Collect news data
    print("Collecting news data...")
    news_data = pd.DataFrame()
    for team in F1_TEAMS.keys():
        team_news = fetch_f1_news(team, start_date, end_date)
        news_data = pd.concat([news_data, team_news])
    
    # Create aggregated news metrics by day/team
    daily_news = news_data.groupby(['date', 'team']).agg(
        news_count=('headline', 'count'),
        avg_sentiment=('sentiment', 'mean'),
        positive_news=('sentiment', lambda x: sum(x > 0.3)),
        negative_news=('sentiment', lambda x: sum(x < -0.3))
    ).reset_index()
    
    # Add 3-day and 7-day rolling averages for news sentiment
    for team in F1_TEAMS.keys():
        team_news = daily_news[daily_news['team'] == team].sort_values('date')
        if len(team_news) > 0:
            team_news['sentiment_3d'] = team_news['avg_sentiment'].rolling(3, min_periods=1).mean()
            team_news['sentiment_7d'] = team_news['avg_sentiment'].rolling(7, min_periods=1).mean()
            
            daily_news.loc[daily_news['team'] == team, 'sentiment_3d'] = team_news['sentiment_3d'].values
            daily_news.loc[daily_news['team'] == team, 'sentiment_7d'] = team_news['sentiment_7d'].values
    
    return {
        'race_results': race_results,
        'stock_data': stock_data,
        'news_data': news_data,
        'daily_news': daily_news
    }


def news_sentiment_analysis(news_data, daily_news):
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='team', y='sentiment', data=news_data)
    plt.title('News Sentiment Distribution by Team')
    plt.xlabel('Team')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    st.image(buffer,use_container_width=True)
    
    
    buffer.close()
    plt.close()
    
    # Visualization 2: Sentiment Trends Over Time
    plt.figure(figsize=(16, 10))
    for i, team in enumerate(F1_TEAMS.keys()):
        team_news = daily_news[daily_news['team'] == team].sort_values('date')
        if len(team_news) > 0:
            plt.subplot(5, 2, i + 1)
            plt.plot(team_news['date'], team_news['sentiment_7d'], label='7-Day Avg')
            plt.title(f'{team} Sentiment Trend')
            plt.xlabel('Date')
            plt.ylabel('Sentiment')
            plt.ylim(-1, 1)
            plt.grid(True)
    
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    
    st.image(buffer,use_container_width=True)
    
    
    buffer.close()
    plt.close()
    
    # Visualization 3: News Volume by Team and Year
    news_data['year'] = news_data['date'].dt.year
    news_volume = news_data.groupby(['team', 'year']).size().reset_index(name='article_count')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='team', y='article_count', hue='year', data=news_volume)
    plt.title('News Volume by Team and Year')
    plt.xlabel('Team')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.legend(title='Year')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    
    st.image(buffer,use_container_width=True)
    
    
    buffer.close()
    plt.close()
    
    return news_volume

def integrated_analysis(race_results, stock_data, daily_news):
   
    # Get teams that have stock data
    teams_with_stock = stock_data['team'].unique()
    
    if len(teams_with_stock) == 0:
        print("No stock data available for integrated analysis")
        return None
    
    # 1. Prepare race data: Get race weekend dates
    race_weekends = race_results[['race_date', 'race_name', 'season']].drop_duplicates()
    race_weekends['race_week_start'] = race_weekends['race_date'] - pd.Timedelta(days=3)
    race_weekends['race_week_end'] = race_weekends['race_date'] + pd.Timedelta(days=3)
    
    # 2. Analyze stock movement around races
    race_impact = []
    
    for _, race in race_weekends.iterrows():
        for team in teams_with_stock:
            # Get team race result
            team_result = race_results[
                (race_results['race_date'] == race['race_date']) & 
                (race_results['team'] == team)
            ]
            
            if len(team_result) == 0:
                continue
                
            # Get stock data before and after race
            pre_race_stock = stock_data[
                (stock_data['team'] == team) & 
                (stock_data['date'] < race['race_date']) & 
                (stock_data['date'] >= race['race_week_start'])
            ].sort_values('date')
            
            post_race_stock = stock_data[
                (stock_data['team'] == team) & 
                (stock_data['date'] > race['race_date']) & 
                (stock_data['date'] <= race['race_week_end'])
            ].sort_values('date')
            
            if len(pre_race_stock) == 0 or len(post_race_stock) == 0:
                continue
            
            # Calculate stock price changes
            pre_race_price = pre_race_stock['Adj Close'].iloc[0]
            post_race_price = post_race_stock['Adj Close'].iloc[-1]
            percent_change = (post_race_price - pre_race_price) / pre_race_price * 100
            
            # Get news sentiment around race
            race_news = daily_news[
                (daily_news['team'] == team) & 
                (daily_news['date'] >= race['race_week_start']) & 
                (daily_news['date'] <= race['race_week_end'])
            ]
            
            avg_sentiment = race_news['avg_sentiment'].mean() if len(race_news) > 0 else None
            news_count = race_news['news_count'].sum() if len(race_news) > 0 else 0
            
            race_impact.append({
                'season': race['season'],
                'race_name': race['race_name'],
                'race_date': race['race_date'],
                'team': team,
                'race_position': team_result['race_position'].iloc[0],
                'points': team_result['points'].iloc[0],
                'pre_race_price': pre_race_price,
                'post_race_price': post_race_price,
                'price_percent_change': percent_change,
                'news_count': news_count,
                'avg_news_sentiment': avg_sentiment
            })
    
    race_impact_df = pd.DataFrame(race_impact)
    
    # Visualization 1: Race Results vs Stock Movement
    plt.figure(figsize=(16, 10))
    for i, team in enumerate(teams_with_stock):
        team_impact = race_impact_df[race_impact_df['team'] == team]
        if len(team_impact) > 0:
            plt.subplot(len(teams_with_stock), 1, i + 1)
            points = team_impact['points']
            pct_change = team_impact['price_percent_change']
            dates = team_impact['race_date']
            
            ax1 = plt.gca()
            ax1.bar(dates, points, alpha=0.7, label='Race Points')
            ax1.set_ylabel('Points', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            ax2.plot(dates, pct_change, 'r-', label='Stock % Change')
            ax2.set_ylabel('Stock % Change', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            plt.title(f'{team} Race Performance vs Stock Movement')
            plt.xlabel('Race Date')
            
            # Add correlation coefficient
            if len(points) > 1:
                corr = np.corrcoef(points, pct_change)[0, 1]
                plt.text(0.02, 0.95, f'Correlation: {corr:.2f}', transform=ax1.transAxes)
    
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    
    st.image(buffer,use_container_width=True)
    
    
    buffer.close()
    plt.close()
    
    # Visualization 2: News Sentiment vs Stock Movement
    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        x='avg_news_sentiment', 
        y='price_percent_change', 
        hue='team', 
        size='points',
        sizes=(20, 200),
        data=race_impact_df.dropna(subset=['avg_news_sentiment'])
    )
    plt.title('Race Week News Sentiment vs Stock Movement')
    plt.xlabel('Average News Sentiment')
    plt.ylabel('Stock Price % Change')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.grid(True)
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    
    st.image(buffer,use_container_width=True)
    
    
    buffer.close()
    plt.close()
    
    # Calculate correlations
    correlations = {}
    for team in teams_with_stock:
        team_data = race_impact_df[race_impact_df['team'] == team].dropna(subset=['avg_news_sentiment'])
        if len(team_data) > 1:
            corr_points_stock = np.corrcoef(team_data['points'], team_data['price_percent_change'])[0, 1]
            corr_sentiment_stock = np.corrcoef(
                team_data['avg_news_sentiment'], 
                team_data['price_percent_change']
            )[0, 1]
            
            correlations[team] = {
                'points_stock_correlation': corr_points_stock,
                'sentiment_stock_correlation': corr_sentiment_stock
            }
    
    return {
        'race_impact': race_impact_df,
        'correlations': correlations
    }

def predictive_modeling(race_results, stock_data, daily_news):
    
    # Get teams that have stock data
    teams_with_stock = stock_data['team'].unique()
    
    if len(teams_with_stock) == 0:
        print("No stock data available for predictive modeling")
        return None
    
    model_results = {}
    
    for team in teams_with_stock:
        print(f"Building predictive model for {team}...")
        
        # Prepare team data
        team_races = race_results[race_results['team'] == team].copy()
        team_stock = stock_data[stock_data['team'] == team].copy()
        team_news = daily_news[daily_news['team'] == team].copy()
        
        # Merge race results with stock data
        merged_data = []
        
        for _, race in team_races.iterrows():
            race_date = race['race_date']
            
            # Get post-race stock data (next 3 trading days)
            post_race_stock = team_stock[
                (team_stock['date'] > race_date) & 
                (team_stock['date'] <= race_date + pd.Timedelta(days=5))
            ].sort_values('date')
            
            if len(post_race_stock) == 0:
                continue
            
            # Calculate stock return after race
            post_race_return = post_race_stock['daily_return'].mean() * 100  # as percentage
            
            # Get pre-race news sentiment (3 days before race)
            pre_race_news = team_news[
                (team_news['date'] >= race_date - pd.Timedelta(days=3)) & 
                (team_news['date'] <= race_date)
            ]
            
            avg_sentiment = pre_race_news['avg_sentiment'].mean() if len(pre_race_news) > 0 else 0
            news_count = pre_race_news['news_count'].sum() if len(pre_race_news) > 0 else 0
            
            # Create feature row
            row = {
                'race_date': race_date,
                'season': race['season'],
                'race_name': race['race_name'],
                'qualifying_position': race['qualifying_position'],
                'race_position': race['race_position'],
                'points': race['points'],
                'fastest_lap': race['fastest_lap'],
                'news_count': news_count,
                'avg_sentiment': avg_sentiment,
                'post_race_return': post_race_return
            }
            
            merged_data.append(row)
        
        if not merged_data:
            print(f"Insufficient data for {team} model")
            continue
            
        model_data = pd.DataFrame(merged_data)
        
        # Features and target
        features = [
            'qualifying_position', 'race_position', 'points', 
            'fastest_lap', 'news_count', 'avg_sentiment'
        ]
        
        X = model_data[features]
        y = model_data['post_race_return']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Train/test split
        if len(X) > 10:  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train a random forest regressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save model results
            model_results[team] = {
                'model': model,
                'feature_importance': feature_importance,
                'mse': mse,
                'r2': r2,
                'test_actual': y_test,
                'test_predicted': y_pred
            }
            
        
    
    return model_results


def analysis() :
    # Set display options
    st.markdown('<div class="main-header">Formula1 News vs Teams Stock Price Analysis</div>', unsafe_allow_html=True)

    
    print(f"Starting F1 Multi-Year Analysis ({START_YEAR}-{END_YEAR})...")

    # Step 1: Collect all data
    all_data = collect_all_data(START_YEAR, END_YEAR, f1_data, F1_TEAMS)
    race_results = all_data['race_results'] 
    stock_data = all_data['stock_data']
    news_data = all_data['news_data']
    daily_news = all_data['daily_news']


    # Step 3: News sentiment analysis
    print("Analyzing news sentiment...")
    news_volume = news_sentiment_analysis(news_data, daily_news)

    # Step 5: Integrated analysis
    print("Performing integrated analysis...")
    integrated_results = integrated_analysis(race_results, stock_data, daily_news)

    # Step 6: Predictive modeling
    print("Building predictive models...")
    model_results = predictive_modeling(race_results, stock_data, daily_news)

analysis()
    
