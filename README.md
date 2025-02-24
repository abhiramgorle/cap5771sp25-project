# F1 Race Analysis Project

This repository contains Jupyter notebooks for analyzing Formula 1 racing data. Follow the instructions below to set up and run the analysis.

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
- scipy

All dependencies are listed in `requirements.txt`.
