# Formula 1 Race Position Performance- Stock Price Analysis

This repository contains Jupyter notebooks, Streamlit applications, and scripts for analyzing Formula 1 racing data and investigating its potential correlation with the stock prices of associated teams.

## Tool Demo Video : https://drive.google.com/file/d/1_HjCiQr3nKtAZYEL0ayahp26TuphUXR9/view?usp=sharing
## Project Presentation Slides: https://docs.google.com/presentation/d/1-fO5r_pu5NGkeg2dAUV2oFpnXWPYnpGiNkVuQjjsQ9c/edit?usp=sharing
## Features & Highlights 
   - *Historical Race Performance*: Podiums, wins, points, and final standings.
   - _Stock Data Retrieval_: Real-time fetching of historical stock prices using yfinance.
   - _Data Alignment_: Matching race days with stock market days before and after races.
   - _Interactive Visualizations_: Time series charts, scatter plots, and box plots.
   - _Statistical Analysis_: Correlations, t-tests for podium vs non-podium finishes.
   - _Machine Learning Model_: Built an ensemble classifier (Voting Classifier with Logistic Regression, Random Forest, Gradient Boosting).
   - Achieved 88.9% accuracy in predicting stock price movement direction after races.


## Prerequisites

Before you begin, ensure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/abhiramgorle/cap5771sp25-project
   cd cap5771sp25-project
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Jupyter Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   This will open a new browser window or tab with the Jupyter interface.

2. **Navigate to the notebook**
   - In the Jupyter interface, navigate to the notebook file (`.ipynb` extension)
   - Click on the notebook to open it

3. **Run the notebook**
   - To run a single cell: Click on the cell and press `Shift + Enter`
   - To run all cells: Click "Kernel" in the top menu, then "Restart & Run All"

## Running the Streamlit Dashboard (Main Tool)
1.) Go to the project folder
   ```bash 
   cd cap5771sp25-project
```
2.) Run the Streamlit app
 ```bash
   streamlit run Final_milestone_3.py
```
3.) Interact with the dashboard!
   - Choose a year and a team from the sidebar.
   - View detailed race performance stats, stock price change visualizations, and machine learning analysis.

## Project Structure

```
cap5771sp25-project/
│
├── Data/                  # Data files
│   ├── circuits.csv
│   ├── drivers.csv
│   ├── races.csv
│   └── ...
│
├── Reports/              # Generated reports
│   └── Milestone1.pdf
│
├── Notebooks/           # Jupyter notebooks
│   └── Milestone1.ipynb
|   |___requirements.txt
│
└── README.md           # This file
```

## Dependencies

The main dependencies for this project are:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- yfinance
- streamlit

All dependencies are listed in `requirements.txt`.
