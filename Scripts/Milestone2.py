import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm

# For modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

# Define F1 teams with stock tickers
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

# Function to load F1 data from Kaggle dataset
def load_f1_data(data_path):
    """
    Load F1 data from the Kaggle dataset
    
    Parameters:
    data_path - Path to the Kaggle dataset
    
    Returns:
    Dictionary with loaded dataframes
    """
    print("Loading F1 data from Kaggle dataset...")
    try:
        # Load main dataframes
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
        
        return {
            'circuits': circuits,
            'constructor_results': constructor_results,
            'constructor_standings': constructor_standings,
            'constructors': constructors,
            'drivers': drivers,
            'races': races,
            'results': results
        }
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    """
    Retrieve historical stock data for a given ticker
    
    Parameters:
    ticker - Stock ticker symbol
    start_date - Start date for data retrieval
    end_date - End date for data retrieval
    
    Returns:
    DataFrame with stock price data
    """
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

# Function to extract constructor performance from F1 data
def extract_constructor_performance(f1_data, constructor_name, year_from, year_to):
    """
    Extract performance data for a specific constructor from F1 dataset
    
    Parameters:
    f1_data - Dictionary with F1 dataframes
    constructor_name - Name of the constructor to extract data for
    year_from - Starting year for data extraction
    year_to - Ending year for data extraction
    
    Returns:
    DataFrame with constructor performance data
    """
    races = f1_data['races']
    constructors = f1_data['constructors']
    constructor_standings = f1_data['constructor_standings']
    results = f1_data['results']
    
    # Filter races within the time range
    races_filtered = races[(races['year'] >= year_from) & (races['year'] <= year_to)]
    
    # Get constructor ID
    constructor_matches = constructors[constructors['name'].str.contains(constructor_name, case=False)]
    if constructor_matches.empty:
        print(f"Constructor '{constructor_name}' not found in the dataset")
        return None
    
    constructor_id = constructor_matches['constructorId'].values[0]
    print(f"Found constructor ID {constructor_id} for {constructor_name}")
    
    # Filter constructor standings for this constructor
    standings = constructor_standings[constructor_standings['constructorId'] == constructor_id]
    
    # Join with races to get dates
    performance_data = pd.merge(
        standings, 
        races_filtered[['raceId', 'date', 'year', 'round', 'name']], 
        on='raceId'
    )
    
    # Get additional race results for this constructor
    constructor_race_results = results[results['constructorId'] == constructor_id]
    
    # Get race results with additional details
    race_results = pd.merge(
        constructor_race_results,
        races_filtered[['raceId', 'date', 'year', 'round', 'name']],
        on='raceId'
    )
    
    # Sort by date
    performance_data = performance_data.sort_values('date')
    performance_data['date'] = pd.to_datetime(performance_data['date'])
    
    race_results = race_results.sort_values('date')
    race_results['date'] = pd.to_datetime(race_results['date'])
    
    print(f"Extracted {len(performance_data)} races for {constructor_name} from {year_from} to {year_to}")
    
    return {
        'standings': performance_data,
        'results': race_results
    }

# Function to align F1 results with stock data
def align_f1_with_stock(performance_data, stock_data, team_name):
    """
    Align F1 race results with stock data - Fixed version that handles Series properly
    
    Parameters:
    performance_data - Dictionary with team performance data
    stock_data - Stock price data
    team_name - Name of the team for identification
    
    Returns:
    DataFrame with aligned F1 and stock data
    """
    print(f"Aligning F1 results with stock data for {team_name}...")
    
    if stock_data is None:
        print(f"No stock data available for {team_name}")
        return None
    
    # Use constructor standings data
    f1_data = performance_data['standings'].copy()
    
    # Convert date columns to datetime
    f1_data['date'] = pd.to_datetime(f1_data['date'])
    stock_data.index = pd.to_datetime(stock_data.index)
    
    # Create a dataframe to store aligned data
    aligned_data = []
    
    for _, race in f1_data.iterrows():
        race_date = race['date']
        
        try:
            # Get the next 5 trading days after the race
            next_trading_days = stock_data.loc[race_date:race_date + timedelta(days=10)].head(5)
            
            # Get the previous 5 trading days before the race
            prev_trading_days = stock_data.loc[race_date - timedelta(days=10):race_date].tail(5)
            
            if not next_trading_days.empty and not prev_trading_days.empty:
                # Calculate stock performance metrics
                pre_race_price = prev_trading_days['Close'].mean()
                post_race_price = next_trading_days['Close'].mean()
                price_change = (post_race_price - pre_race_price) / pre_race_price * 100
                
                # Create a row with race and stock data
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
# Function to visualize F1 performance vs stock price changes
def visualize_performance_vs_stock(aligned_data, constructor_name):
    """
    Create visualizations of F1 performance vs stock price changes
    Ultra-robust version avoiding any Series operations
    
    Parameters:
    aligned_data - DataFrame with aligned F1 and stock data
    constructor_name - Constructor name for the chart title
    
    Returns:
    None (displays plots)
    """
    if aligned_data is None or len(aligned_data) == 0:
        print(f"No data to visualize for {constructor_name}")
        return
    
    
    try:
        dates = [d for d in aligned_data['race_date']]
        positions = [float(p) if isinstance(p, (int, float)) else float('nan') for p in aligned_data['position']]
        price_changes = [float(pc.iloc[0]) if isinstance(pc, pd.Series) and not pc.empty else float('nan') for pc in aligned_data['price_change_pct']]
        years = [int(y) if isinstance(y, (int, float)) else 0 for y in aligned_data['year']]
        rounds = [int(r) if isinstance(r, (int, float)) else 0 for r in aligned_data['round']]
    except Exception as e:
        print(f"Error converting data: {e}")
        # Print for debugging
        print("Column names:", aligned_data.columns.tolist())
        print("Data types:", {col: aligned_data[col].dtype for col in aligned_data.columns})
        return
    
    # Create figure with two y-axes for time series plot
    try:
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        
        # Plot positions (inverted so better positions are higher)
        ax1.scatter(dates, [-p for p in positions], color='red', s=80)
        ax1.plot(dates, [-p for p in positions], color='red', alpha=0.6)
        
        ax1.set_xlabel('Race Date')
        ax1.set_ylabel('Race Position (inverted)', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        
        # Set y-ticks for positions
        max_pos = max([p for p in positions if not np.isnan(p)], default=10)
        ax1.set_yticks([-i for i in range(1, int(max_pos) + 1)])
        ax1.set_yticklabels([i for i in range(1, int(max_pos) + 1)])
        
        # Create second y-axis
        ax2 = ax1.twinx()
        
        # Plot bars for price changes with manually assigned colors
        for i, (date, change) in enumerate(zip(dates, price_changes)):
            if np.isnan(change):
                continue
            color = 'green' if change >= 0 else 'red'
            ax2.bar(date, change, color=color, alpha=0.5, width=10)  # width in days
        
        ax2.set_ylabel('Stock Price Change (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title(f'{constructor_name} F1 Performance vs Stock Price Changes')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error creating time series plot: {e}")
    
    # Create scatter plot
    try:
        # Filter out NaN values
        valid_data = [(p, c) for p, c in zip(positions, price_changes) if not np.isnan(p) and not np.isnan(c)]
        if not valid_data:
            print("No valid data points for scatter plot")
            return
            
        valid_positions, valid_changes = zip(*valid_data)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_positions, valid_changes, s=100)
        
        # Calculate trend line if we have enough points
        if len(valid_positions) > 1:
            z = np.polyfit(valid_positions, valid_changes, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(valid_positions), max(valid_positions), 100)
            plt.plot(x_range, p(x_range), "r--")
            
            # Calculate correlation
            correlation = np.corrcoef(valid_positions, valid_changes)[0, 1]
            plt.annotate(f'Correlation: {correlation:.4f}', 
                        xy=(0.05, 0.95), xycoords='axes fraction', 
                        fontsize=12)
        
        plt.title(f'Relationship Between {constructor_name} Race Position and Stock Price Change')
        plt.xlabel('Race Position (lower is better)')
        plt.ylabel('Stock Price Change (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
    
    # Create boxplot for podium vs non-podium
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
            
            # Create data for boxplot
            box_data = [podium_changes, non_podium_changes]
            plt.boxplot(box_data, labels=['Podium', 'Non-Podium'])
            
            plt.title(f'{constructor_name} Stock Price Changes: Podium vs Non-Podium Finishes')
            plt.ylabel('Price Change (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough data for podium comparison boxplot")
    except Exception as e:
        print(f"Error creating boxplot: {e}")
# Function to analyze the statistical relationship between F1 performance and stock changes
def analyze_performance_stock_relationship(aligned_data, constructor_name):
    """
    Analyze statistical relationship between F1 performance and stock price changes
    Ultra-robust version avoiding Series operations
    
    Parameters:
    aligned_data - DataFrame with aligned F1 and stock data
    constructor_name - Constructor name for analysis identification
    
    Returns:
    Dictionary with analysis results
    """
    print(f"\n===== {constructor_name} F1 Performance and Stock Analysis =====")
    
    if aligned_data is None or len(aligned_data) == 0:
        print(f"No data to analyze for {constructor_name}")
        return None
    
    try:
        # Convert DataFrame columns to Python lists for robustness
        positions = []
        price_changes = []
        
        # Handle each row individually to avoid Series operations
        for _, row in aligned_data.iterrows():
            try:
                pos = float(row['position'])
                change = float(row['price_change_pct'])
                
                # Skip if either value is NaN
                if not np.isnan(pos) and not np.isnan(change):
                    positions.append(pos)
                    price_changes.append(change)
            except (ValueError, TypeError):
                continue  # Skip rows with conversion errors
        
        if not positions or not price_changes:
            print(f"No valid numeric data to analyze for {constructor_name}")
            return None
            
        # Calculate correlation using numpy to avoid pandas Series operations
        correlation = np.corrcoef(positions, price_changes)[0, 1]
        print(f"Correlation between position and stock change: {correlation:.4f}")
        
        # Separate podium and non-podium results
        podium_changes = [change for pos, change in zip(positions, price_changes) if pos <= 3]
        non_podium_changes = [change for pos, change in zip(positions, price_changes) if pos > 3]
        
        # Calculate average changes
        avg_change_podium = np.mean(podium_changes) if podium_changes else None
        avg_change_non_podium = np.mean(non_podium_changes) if non_podium_changes else None
        
        if avg_change_podium is not None:
            print(f"Average stock change after podium finishes: {avg_change_podium:.2f}%")
        if avg_change_non_podium is not None:
            print(f"Average stock change after non-podium finishes: {avg_change_non_podium:.2f}%")
        
        # Separate win and non-win results
        victory_changes = [change for pos, change in zip(positions, price_changes) if pos == 1]
        non_victory_changes = [change for pos, change in zip(positions, price_changes) if pos != 1]
        
        # Calculate average changes for wins/non-wins
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
        
        # Return results as dictionary
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
        # Print column info for debugging
        print("DataFrame columns:", aligned_data.columns.tolist())
        print("Column types:", {col: str(aligned_data[col].dtype) for col in aligned_data.columns})
        return None
# Function to build predictive model for stock price changes based on F1 performance
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


# Function to predict stock price changes for future races
def predict_future_stock_changes(model_info, upcoming_race_data):
    """
    Predict stock price changes for upcoming races
    
    Parameters:
    model_info - Dictionary with model information
    upcoming_race_data - DataFrame with upcoming race information
    
    Returns:
    DataFrame with predicted stock price changes
    """
    if model_info is None:
        print("No model information available")
        return None
    
    # Extract model components
    model = model_info['best_model']['model']
    scaler = model_info['best_model']['scaler']
    features = model_info['best_model']['features']
    
    # Prepare features for prediction
    X_pred = upcoming_race_data[features]
    
    # Scale features
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make predictions
    predictions = model.predict(X_pred_scaled)
    
    # Add predictions to the data
    results = upcoming_race_data.copy()
    results['predicted_price_change'] = predictions
    
    return results


# Main execution function
def main():
    data_path = "Data"  
    
    # Load F1 data
    f1_data = load_f1_data(data_path)
    if f1_data is None:
        print("Failed to load F1 data. Exiting.")
        return
    
    # Define analysis parameters
    start_year = 2015  # Ferrari IPO was in October 2015
    end_year = 2023
    
    # Define F1 teams with stock tickers to analyze
    public_teams = {
        'Ferrari': 'RACE',  # Most direct connection
    }
    
    # Dictionary to store results
    team_results = {}
    
    for team, ticker in public_teams.items():
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        print(f"\nProcessing {team} ({ticker}) from {start_year} to {end_year}")
        
        # 1. Get stock data
        print(f"Getting stock data for {ticker}...")
        stock_data = get_stock_data(ticker, start_date, end_date)
        
        if stock_data is None:
            continue
        
        # 2. Extract team performance data
        print(f"Extracting {team} performance data...")
        performance_data = extract_constructor_performance(f1_data, team, start_year, end_year)
        
        if performance_data is None:
            continue
        
        # 3. Align F1 results with stock data
        print(f"Aligning F1 results with stock data for {team}...")
        aligned_data = align_f1_with_stock(performance_data, stock_data, team)
        
        if aligned_data is None:
            continue
        
        # 4. Visualize relationship
        print(f"Creating visualizations for {team}...")
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
    
    # 8. Summarize findings
    print("\n===== Summary of Findings =====")
    for team, results in team_results.items():
        if 'analysis_results' in results and results['analysis_results'] is not None:
            analysis = results['analysis_results']
            print(f"\n{team}:")
            print(f"- Sample size: {analysis['sample_size']} races")
            print(f"- Position-Stock Change Correlation: {analysis['position_correlation']:.4f}")
            
            if analysis['avg_change_podium'] is not None and analysis['avg_change_non_podium'] is not None:
                print(f"- Avg Stock Change after Podium: {analysis['avg_change_podium']:.2f}%")
                print(f"- Avg Stock Change after Non-Podium: {analysis['avg_change_non_podium']:.2f}%")
                
            if analysis['significant_difference']:
                print(f"- The difference between podium and non-podium stock changes is statistically significant (p={analysis['p_value']:.4f})")
            
            

    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()