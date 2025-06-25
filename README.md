# UFC Outcome Prediction

## Project Overview

This machine learning project aims to predict UFC fight outcomes by analyzing historical fighter data, fight statistics, and betting odds. The system combines data from multiple sources, processes it through a pipeline, and uses ML classification algorithms to make predictions.

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Pipeline Steps](#pipeline-steps)
- [Contributing](#contributing)

## Setup

### Prerequisites

- Python 3.8+
- Virtual Environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd UFC_Outcome_Prediction
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
UFC_Outcome_Prediction/
├── data/                      # Data directory (gitignored)
│   ├── raw/                  # Raw scraped data
│   │   ├── events/          # Event-related data
│   │   ├── events_with_odds/# Event data with betting odds
│   │   └── fighters/        # Fighter-related data
│   ├── processed/           # Cleaned and processed data
│   └── aggregated/          # Feature engineered data
├── src/                      # Source code
│   ├── data/                # Data processing modules
│   │   ├── data_cleaner.py    # Data cleaning utilities
│   │   ├── events_scraper.py  # Event data collection
│   │   ├── feature_engineer.py# Feature engineering
│   │   ├── fighters_scraper.py# Fighter data collection
│   │   └── odds_scraper.py    # Betting odds collection
│   └── utils/               # Utility functions
│       └── logging_config.py  # Logging configuration
├── notebooks/               # Jupyter notebooks
│   ├── Inference.ipynb       # Basic inference notebook
│   └── Inference_with_odds.ipynb # Inference with odds data
└── step{1-5}_*.py          # Pipeline execution scripts
```

## Pipeline Steps

The data processing pipeline consists of several sequential steps, each implemented as a separate script:

1. `step1_scrape_fighters.py`: Collects fighter data including basic information, fight history, and statistics
2. `step2_scrape_events.py`: Gathers event details and fight cards
3. `step3_scrape_odds.py`: Collects betting odds data from various sources
4. `step4_clean.py`: Cleans and standardizes the collected data
5. `step5_feature_engineering.py`: Generates features for the prediction model

### Inference

Two Jupyter notebooks are provided for making predictions:
- `Inference.ipynb`: Basic prediction using fighter statistics
- `Inference_with_odds.ipynb`: Enhanced prediction incorporating betting odds

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 