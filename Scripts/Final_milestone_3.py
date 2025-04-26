import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample
import os
import streamlit as st
from io import BytesIO

def get_stock_data(ticker, start_date, end_date):
    #To Retrieve historical stock data for a given team's ticker#
    if ticker is None:
        return None
    
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data found for ticker {ticker}")
            return None
        
        print(f"Successfully downloaded stock data for {ticker} ({len(stock_data)} trading days)")
        return stock_data
    except Exception as e:
        print(f"Error retrieving stock data for {ticker}: {str(e)}")
        return None

def display_team_stats(year, team, f1_data):
    ## Display team statistics for a given year and team
    circuits = f1_data['circuits']
    constructor_results = f1_data['constructor_results']
    constructor_standings = f1_data['constructor_standings']
    constructors = f1_data['constructors']
    drivers = f1_data['drivers']
    races = f1_data['races']
    results = f1_data['results']

    # Get constructorId for the team
    team_row = constructors[constructors['name'].str.lower() == team.lower()]
    if team_row.empty:
        st.error(f"Team '{team}' not found!")
        return

    constructor_id = team_row.iloc[0]['constructorId']
    races_in_year = races[races['year'] == year]
    results_in_year = results[results['raceId'].isin(races_in_year['raceId'])]
    team_results = results_in_year[results_in_year['constructorId'] == constructor_id]

    # The Proooo Stats
    total_podiums = len(team_results[team_results['positionOrder'].isin([1, 2, 3])])
    total_wins = len(team_results[team_results['positionOrder'] == 1])

    standings_in_year = constructor_standings[
        (constructor_standings['raceId'].isin(races_in_year['raceId'])) &
        (constructor_standings['constructorId'] == constructor_id)
    ]

    if standings_in_year.empty:
        total_points = 0
        final_position = None
    else:
        latest_race = standings_in_year.sort_values('raceId', ascending=False).iloc[0]
        total_points = latest_race['points']
        final_position = latest_race['position']

    # CSS-style card block for the gray background
    card_style = '''
        <style>
        .card {
            background-color: #e0e4ea;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .card h4 {
            margin: 0.5rem 0 0.2rem 0;
            font-size: 1.2rem;
        }
        .card p {
            margin: 0;
            font-size: 1.5rem;
            font-weight: bold;
        }
        </style>
    '''
    st.markdown(card_style, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Podiums and Wins")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
                <div class='card'>
                    <h4>Total Podiums</h4>
                    <p>{total_podiums}</p>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class='card'>
                    <h4>Total Wins</h4>
                    <p>{total_wins}</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.subheader("🏁 Constructor Standings")
        c3, c4 = st.columns(2)
        with c3:
            st.markdown(f"""
                <div class='card'>
                    <h4>Total Points</h4>
                    <p>{total_points}</p>
                </div>
            """, unsafe_allow_html=True)
        with c4:
            if final_position is not None:
                st.markdown(f"""
                    <div class='card'>
                        <h4>Final Position</h4>
                        <p>{final_position}</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='card'>
                        <h4>Final Position</h4>
                        <p>Not Available</p>
                    </div>
                """, unsafe_allow_html=True)

def extract_constructor_performance(f1_data, constructor_name, year_from, year_to):
    #to Extract performance data for a specific constructor team from the F1 dataset#
    races = f1_data['races']
    constructors = f1_data['constructors']
    constructor_standings = f1_data['constructor_standings']
    results = f1_data['results']
    
    # Filter races within the time range
    races['year'] = pd.to_numeric(races['year'], errors='coerce') 
    races_filtered = races[(races['year'] >= int(year_from)) & (races['year'] <= int(year_to))]
    
    # to Get constructor ID
    constructor_matches = constructors[constructors['name'].str.contains(constructor_name, case=False)]
    if constructor_matches.empty:
        print(f"Constructor '{constructor_name}' not found in the dataset")
        return None
    
    constructor_id = constructor_matches['constructorId'].values[0]
    print(f"Found constructor ID {constructor_id} for {constructor_name}")
    
 
    standings = constructor_standings[constructor_standings['constructorId'] == constructor_id]
    
   
    performance_data = pd.merge(
        standings, 
        races_filtered[['raceId', 'date', 'year', 'round', 'name']], 
        on='raceId'
    )
    
    constructor_race_results = results[results['constructorId'] == constructor_id]
    
    race_results = pd.merge(
        constructor_race_results,
        races_filtered[['raceId', 'date', 'year', 'round', 'name']],
        on='raceId'
    )
    
    # Sorting by the date
    performance_data = performance_data.sort_values('date')
    performance_data['date'] = pd.to_datetime(performance_data['date'])
    
    race_results = race_results.sort_values('date')
    race_results['date'] = pd.to_datetime(race_results['date'])
    
    print(f"Extracted {len(performance_data)} races for {constructor_name} from {year_from} to {year_to}")
    
    return {
        'standings': performance_data,
        'results': race_results
    }

def align_f1_with_stock(performance_data, stock_data, team_name):
    #to Align F1 race results with the stock data#
    print(f"Aligning F1 results with stock data for {team_name}...")
    
    if stock_data is None:
        print(f"No stock data available for {team_name}")
        return None
    
    # Use the constructor standings data
    f1_data = performance_data['standings'].copy()
    
    # Convert the date columns to datetime
    f1_data['date'] = pd.to_datetime(f1_data['date'])
    stock_data.index = pd.to_datetime(stock_data.index)
    
    
    aligned_data = []
    
    for _, race in f1_data.iterrows():
        race_date = race['date']
        
        try:
            # Get the next 5 trading days after the race
            next_trading_days = stock_data.loc[race_date:race_date + timedelta(days=10)].head(2)
            
            # Get the previous 5 trading days before the race
            prev_trading_days = stock_data.loc[race_date - timedelta(days=10):race_date].tail(2)
            
            if not next_trading_days.empty and not prev_trading_days.empty:
                # to Calculate stock performance metrics
                pre_race_price = prev_trading_days['Close'].mean()
                post_race_price = next_trading_days['Close'].mean()
                price_change = (post_race_price - pre_race_price) / pre_race_price * 100
                
                # to Create a row with race and stock data
                row = {
                    'race_id': race['raceId'],
                    'race_name': race['name'],
                    'race_date': race_date,
                    'year': race['year'],
                    'round': race['round'],
                    'points': race['points'],
                    'position': race['position'],
                    'pre_race_price': pre_race_price,
                    'post_race_price': post_race_price,
                    'price_change_pct': price_change,
                    'wins': race['wins'],
                    'team': team_name,
                }
                aligned_data.append(row)
        except Exception as e:
            print(f"Error processing race {race['name']} {race['year']}: {str(e)}")
            continue
    
    if not aligned_data:
        print(f"No aligned data available for {team_name}")
        return None
    
    return pd.DataFrame(aligned_data)

def visualize_performance_vs_stock(aligned_data, constructor_name):
    #to Create visualizations of F1 performance vs stock price changes for the teams"
    if aligned_data is None or len(aligned_data) == 0:
        print(f"No data to visualize for {constructor_name}")
        return
    
    
    try:
        dates = [d for d in aligned_data['race_date']]
        # print("Dates:",dates)
        positions = [float(p) if isinstance(p, (int, float)) else float('nan') for p in aligned_data['position']]
        # print("Positons:",positions)
        # print(aligned_data['price_change_pct'])
        price_changes = [float(pc.iloc[0]) if isinstance(pc, pd.Series) and not pc.empty else float('nan') for pc in aligned_data['price_change_pct']]
        # print("Price changes:",price_changes)
        years = [int(y) if isinstance(y, (int, float)) else 0 for y in aligned_data['year']]
        rounds = [int(r) if isinstance(r, (int, float)) else 0 for r in aligned_data['round']]
    except Exception as e:
        print(f"Error converting data: {e}")
        # Printing for debugg
        print("Column names:", aligned_data.columns.tolist())
        print("Data types:", {col: aligned_data[col].dtype for col in aligned_data.columns})
        return
    
    #to  Create figure with two y-axes for time series plot
    try:
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        
        
        ax1.scatter(dates, [-p for p in positions], color='red', s=80)
        ax1.plot(dates, [-p for p in positions], color='red', alpha=0.6)
        
        ax1.set_xlabel('Race Date')
        ax1.set_ylabel('Race Position (inverted)', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        
        
        max_pos = max([p for p in positions if not np.isnan(p)], default=10)
        ax1.set_yticks([-i for i in range(1, int(max_pos) + 1)])
        ax1.set_yticklabels([i for i in range(1, int(max_pos) + 1)])
        
        # to Create second y-axis
        ax2 = ax1.twinx()
        
        # Plot bars for the price schages
        for i, (date, change) in enumerate(zip(dates, price_changes)):
            if np.isnan(change):
                continue
            color = 'green' if change >= 0 else 'red'
            ax2.bar(date, change, color=color, alpha=0.5, width=10)  
        
        ax2.set_ylabel('Stock Price Change (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f'{constructor_name} F1 Performance vs Stock Price Changes')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        st.image(buffer,use_container_width=True)
        
        buffer.close()
        plt.close()
    except Exception as e:
        print(f"Error creating time series plot: {e}")
    
    # to Create scatter plot
    try:
        valid_data = [(p, c) for p, c in zip(positions, price_changes) if not np.isnan(p) and not np.isnan(c)]
        if not valid_data:
            print("No valid data points for scatter plot")
            return
            
        valid_positions, valid_changes = zip(*valid_data)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_positions, valid_changes, s=100)
        
        if len(valid_positions) > 1:
            z = np.polyfit(valid_positions, valid_changes, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(valid_positions), max(valid_positions), 100)
            plt.plot(x_range, p(x_range), "r--")
            
            correlation = np.corrcoef(valid_positions, valid_changes)[0, 1]
            plt.annotate(f'Correlation: {abs(correlation):.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction', 
                        fontsize=12)
        
        plt.title(f'Relationship Between {constructor_name} Race Position and Stock Price Change')
        plt.xlabel('Race Position (lower is better)')
        plt.ylabel('Stock Price Change (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        st.image(buffer,use_container_width=True)
        
        buffer.close()
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
    
    # to  Create boxplot for podium vs non-podium
    try:
        podium_changes = []
        non_podium_changes = []
        
        for pos, change in zip(positions, price_changes):
            if np.isnan(pos) or np.isnan(change):
                continue
            if pos <= 3:
                podium_changes.append(change)
            else:
                non_podium_changes.append(change)
        
        if podium_changes and non_podium_changes:
            plt.figure(figsize=(10, 6))
            
            box_data = [podium_changes, non_podium_changes]
            plt.boxplot(box_data, labels=['Podium', 'Non-Podium'])
            
            plt.title(f'{constructor_name} Stock Price Changes: Podium vs Non-Podium Finishes')
            plt.ylabel('Price Change (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            
            st.image(buffer,use_container_width=True)
            
            buffer.close()
            plt.close()
        else:
            print("Not enough data for podium comparison boxplot")
    except Exception as e:
        print(f"Error creating boxplot: {e}")

def analyze_performance_stock_relationship(aligned_data, constructor_name):
    # to Analyze statistical relations between F1 performance and stock price changes of the teams#
    print(f"\n===== {constructor_name} F1 Performance and Stock Analysis =====")
    
    if aligned_data is None or len(aligned_data) == 0:
        print(f"No data to analyze for {constructor_name}")
        return None
    
    try:
        positions = []
        price_changes = []
        
        for _, row in aligned_data.iterrows():
            try:
                pos = float(row['position'])
                change = float(row['price_change_pct'])
                
                if not np.isnan(pos) and not np.isnan(change):
                    positions.append(pos)
                    price_changes.append(change)
            except (ValueError, TypeError):
                continue 
        
        if not positions or not price_changes:
            print(f"No valid numeric data to analyze for {constructor_name}")
            return None
            
        correlation = np.corrcoef(positions, price_changes)[0, 1]
        print(f"Correlation between position and stock change: {correlation:.4f}")
        
        podium_changes = [change for pos, change in zip(positions, price_changes) if pos <= 3]
        non_podium_changes = [change for pos, change in zip(positions, price_changes) if pos > 3]
        
        avg_change_podium = np.mean(podium_changes) if podium_changes else None
        avg_change_non_podium = np.mean(non_podium_changes) if non_podium_changes else None
        
        if avg_change_podium is not None:
            print(f"Average stock change after podium finishes: {avg_change_podium:.2f}%")
        if avg_change_non_podium is not None:
            print(f"Average stock change after non-podium finishes: {avg_change_non_podium:.2f}%")
        
        victory_changes = [change for pos, change in zip(positions, price_changes) if pos == 1]
        non_victory_changes = [change for pos, change in zip(positions, price_changes) if pos != 1]
        
        avg_change_victory = np.mean(victory_changes) if victory_changes else None
        avg_change_non_victory = np.mean(non_victory_changes) if non_victory_changes else None
        
        if avg_change_victory is not None:
            print(f"Average stock change after victories: {avg_change_victory:.2f}%")
        if avg_change_non_victory is not None:
            print(f"Average stock change after non-victories: {avg_change_non_victory:.2f}%")
        
        t_stat = None
        p_value = None
        significant_difference = False
        
        if len(podium_changes) > 1 and len(non_podium_changes) > 1:
            from scipy.stats import ttest_ind


            try:
                t_stat, p_value = ttest_ind(
                    podium_changes, 
                    non_podium_changes,
                    equal_var=False
                )
                significant_difference = p_value < 0.05
                
                print(f"T-test results: t={t_stat:.4f}, p={p_value:.4f}")
                print(f"Statistically significant difference: {significant_difference}")
            except Exception as e:
                print(f"Error performing t-test: {e}")
        
        results = {
            'constructor': constructor_name,
            'position_correlation': correlation,
            'avg_change_podium': avg_change_podium,
            'avg_change_non_podium': avg_change_non_podium,
            'avg_change_victory': avg_change_victory,
            'avg_change_non_victory': avg_change_non_victory,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_difference': significant_difference,
            'sample_size': len(positions)
        }
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {constructor_name} data: {e}")
        print("DataFrame columns:", aligned_data.columns.tolist())
        print("Column types:", {col: str(aligned_data[col].dtype) for col in aligned_data.columns})
        return None

def build_stock_direction_model(aligned_data, constructor_name):
    print(f"\nBuilding ensemble classification model for {constructor_name} stock direction...")

    if aligned_data is None or len(aligned_data) < 10:
        print(f"Not enough data to build a model for {constructor_name}")
        return None

    # 1. Build clean feature-label dataset
    data_rows = []
    for _, row in aligned_data.iterrows():
        try:
            position = float(row['position'])
            points = float(row.get('points', 0.0))
            race_round = float(row.get('round', 0.0))
            price_change = float(row['price_change_pct'])

            if abs(price_change) > 10:
                continue

            label = 1 if price_change > 0 else 0

            data_rows.append({
                'position': position,
                'points': points,
                'round': race_round,
                'price_up': label
            })
        except:
            continue

    if len(data_rows) < 10:
        print("Not enough valid data after filtering.")
        return None

    df = pd.DataFrame(data_rows)

    # 2. Balance the classes
    class_0 = df[df['price_up'] == 0]
    class_1 = df[df['price_up'] == 1]
    if len(class_0) > 0 and len(class_1) > 0:
        majority = class_0 if len(class_0) > len(class_1) else class_1
        minority = class_1 if len(class_0) > len(class_1) else class_0
        minority_upsampled = resample(minority, 
                                      replace=True, 
                                      n_samples=len(majority), 
                                      random_state=42)
        df = pd.concat([majority, minority_upsampled])

    # 3. Prepare features
    features = ['position', 'points', 'round']
    X = df[features]
    y = df['price_up']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Define individual models
    logreg = LogisticRegression(class_weight='balanced', random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    # 5. Voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('logreg', logreg),
            ('rf', rf),
            ('gb', gb)
        ],
        voting='soft'  # use probabilities
    )

    ensemble.fit(X_train_scaled, y_train)
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    r2 = r2_score(y_test, y_proba)

    mae = mean_absolute_error(y_test, y_proba)
    mse = mean_squared_error(y_test, y_proba)

    


    print(f"\n===== Ensemble Model Results =====")
    print(f"Accuracy:   {acc:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"ROC AUC:    {auc:.4f}")
    print(f"R2 Score:   {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Sample Size: {len(X_test)}")
    print(f"Sample Size: {len(df)} (balanced)")

    return {
        'model': ensemble,
        'scaler': scaler,
        'features': features
    }


       
if __name__ == "__main__":
    st.set_page_config(
    page_title="Formula1 vs Team's Stock Price Analysis",
    layout="wide",
    initial_sidebar_state="expanded"  
    )
    data_path = "../Data"  # Path to the Data folder
    
    
    circuits = pd.read_csv(f"{data_path}/circuits.csv")
    constructor_results = pd.read_csv(f"{data_path}/constructor_results.csv")
    constructor_standings = pd.read_csv(f"{data_path}/constructor_standings.csv")
    constructors = pd.read_csv(f"{data_path}/constructors.csv")
    drivers = pd.read_csv(f"{data_path}/drivers.csv")
    races = pd.read_csv(f"{data_path}/races.csv")
    results = pd.read_csv(f"{data_path}/results.csv")
    
    f1_data =  {
        'circuits': circuits,
        'constructor_results': constructor_results,
        'constructor_standings': constructor_standings,
        'constructors': constructors,
        'drivers': drivers,
        'races': races,
        'results': results
    }
    
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

    with st.sidebar:
        st.title("Select Filters")
        
        # Racing Year selection
        st.header("Select Year")
        year = st.radio("", options=["2024","2023", "2022","2021","2020", "2019","2018","2017","2016",], index=0)

        
        # Racing Company selection
        st.header("Select a Team")
        company = st.selectbox("", options=["Ferrari","Mercedes","Aston Martin","Red Bull"], index=0)
        st.header("   ")
        st.link_button(label="News Sentiment Analysis", url="http://localhost:8501/",  icon=None, disabled=False, use_container_width=False)
        st.header("   ")
        with st.expander("About Formula 1"):
            st.write("""
            Formula 1 (F1) is a world championship series that features the fastest road course racing cars, 
            driven by the best drivers in the world.
            The sport has a rich history, with iconic teams and drivers, and it continues to evolve
            with advancements in technology and sustainability initiatives. 
            The F1 calendar consists of multiple Races held on different circuits around the world,
            each contributing to the championship standings.
                """)
        
        with st.expander("Data Source"):
                st.write("""
                Data is sourced from Kaggle DataSet : https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020.
                """)

    start_year =year  
    end_year = year

    # Define F1 teams with stock tickers to analyze
    public_teams1 = {
    "Ferrari": "RACE",
    "Mercedes": "MBGAF",
    "Aston Martin": "AML.L",
    "Red Bull": "ORCL",
    }
    public_teams = {
    "Ferrari": ["RACE", "HPQ"],
    "Red Bull": ["ORCL", "MC.PA"],
    "Mercedes": ["MBGAF", "ADDYY", "SAP", "TMV.DE"],
    "Aston Martin": ["AML.L", "2222.SR", "PUM.DE", "COIN", "CTSH"],
    }

    stock_tickers = {
    'Ferrari': 'RACE',          # Ferrari N.V. (NYSE)
    'Mercedes': 'MBGAF',        # Mercedes-Benz Group AG (OTC)
    'Aston Martin': 'AML.L',    # Aston Martin Lagonda (London Stock Exchange)
    'Alpine': 'RNSDF',          # Renault (OTC)
    }

    # Map constructor names in the dataset to standard names
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
    st.markdown('<div class="main-header">Formula1 vs Teams Stock Price Analysis</div>', unsafe_allow_html=True)

    
    team_results = {}
    team = company
    ticker = public_teams1[company]
    try :
            st.markdown(f"## {year} Formula-1 Report for {company}")
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            display_team_stats(int(start_year), company, f1_data)
        
            stock_data = get_stock_data(ticker, start_date, end_date)
            
            
            
            # 2. Extract team performance data 
            performance_data = extract_constructor_performance(f1_data, team, start_year, end_year)
            
            
            # 3. Align F1 results with stock data 
            aligned_data = align_f1_with_stock(performance_data, stock_data, team)
            
            
            # 4. Visualize relationship between F1 performance and stock price changes
            visualize_performance_vs_stock(aligned_data, team)
            
            # 5. Statistical analysis
            analysis_results = analyze_performance_stock_relationship(aligned_data, team)
            
            # 6. Build predictive model
            model_results = build_stock_direction_model(aligned_data, team)
            
            # 7. Store results
            team_results[team] = {
                'aligned_data': aligned_data,
                'analysis_results': analysis_results,
                'model_results': model_results
            }
            print(team_results[team]["model_results"])

    except Exception as e:
        print(f"Error processing {team}: {e}")
        st.error(f"Error processing {team}: {e}")