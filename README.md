# UFC Fight Prediction System

A comprehensive data pipeline and machine learning system for predicting UFC fight outcomes using historical fighter statistics and betting odds data.

## Project Overview

This project aims to predict UFC fight outcomes by analyzing historical fighter data, fight statistics, and betting odds. The system combines data from multiple sources, processes it through a sophisticated pipeline, and uses machine learning to make predictions.

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Data Processing](#data-processing)
- [Contributing](#contributing)

## Setup

### Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MMA_aggregated_try2
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
MMA_aggregated_try2/
├── data/                      # Data directory
│   ├── raw/                  # Raw scraped data
│   ├── processed/            # Cleaned and processed data
│   └── aggregated/           # Feature engineered data
├── src/                      # Source code
│   ├── data/                # Data processing modules
│   ├── models/              # Model training and prediction
│   └── utils/               # Utility functions
├── notebooks/               # Jupyter notebooks
├── logs/                    # Application logs
└── models/                  # Saved model files
```

## Data Collection

The system collects data from multiple sources:

### Fighter Data
- Basic fighter information
- Fight history
- Performance statistics
- Physical attributes

### Event Data
- Event details
- Fight cards
- Results

### Betting Odds
- Historical betting odds
- Odds movement data
- Multiple bookmaker data

## Data Processing

The data processing pipeline consists of several stages:

### 1. Data Cleaning
- Handling missing values
- Standardizing formats
- Removing duplicates
- Validating data consistency

### 2. Feature Engineering
- Creating fighter performance metrics
- Computing historical statistics
- Generating fight-specific features
- Incorporating betting odds data

### 3. Data Aggregation
- Combining fighter statistics
- Merging betting odds
- Creating final feature set

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 