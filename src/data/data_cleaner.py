import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from pathlib import Path
from ..utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d")
        self.output_dir = Path(f"data/processed/{self.timestamp}")
        
    def _height_to_cm(self, height_str):
        """Convert height from ft'in" format to centimeters"""
        try:
            feet, inches = map(int, height_str.replace('"', '').split("' "))
            total_inches = feet * 12 + inches
            cm = total_inches * 2.54
            return int(cm)
        except (ValueError, TypeError):
            logger.warning(f"Invalid height format: {height_str}")
            return None

    def _weight_to_kg(self, weight_str):
        """Convert weight from lbs to kg"""
        try:
            lbs = int(weight_str.split()[0])
            kg = round(lbs * 0.453592)
            return int(kg)
        except (ValueError, TypeError):
            logger.warning(f"Invalid weight format: {weight_str}")
            return None

    def _inches_to_cm(self, height_str):
        """Convert reach from inches to centimeters"""
        try:
            inches = int(height_str.replace('"', ''))
            cm = round(inches * 2.54)
            return int(cm)
        except (ValueError, TypeError):
            logger.warning(f"Invalid reach format: {height_str}")
            return None

    def _clean_fighters_data(self, fighters_data):
        """Clean and preprocess fighters data"""
        logger.info("Cleaning fighters data...")
        
        # Make a copy to avoid modifying the original
        df = fighters_data.copy()
        
        # Convert column names to lowercase
        df.columns = map(str.lower, df.columns)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed duplicates. {len(fighters_data) - len(df)} duplicate entries found")
        
        # Convert sex to binary
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
        
        # Convert and clean height data
        df['height'] = df['height'].apply(self._height_to_cm)
        df['height'] = df['height'].fillna(df.groupby('sex')['height'].transform('median'))
        
        # Convert and clean weight data
        df['weight'] = df['weight'].apply(self._weight_to_kg)
        df['weight'] = df['weight'].fillna(df.groupby('sex')['weight'].transform('median'))
        
        # Convert and clean reach data
        df['reach'] = df['reach'].apply(self._inches_to_cm)
        df['reach'] = df['reach'].fillna(df.groupby('sex')['reach'].transform('median'))
        
        # Clean stance data
        df['stance'] = df['stance'].replace('', df['stance'].mode()[0])
        
        # Clean date of birth data
        df['dob'] = df['dob'].replace('--', np.nan)
        
        # Clean country names
        df['country'] = df['country'].str.replace(' ', '_')
        
        logger.info("Fighters data cleaning completed")
        return df

    def _clean_events_data(self, events_data, has_odds=False):
        """Clean and preprocess events data"""
        logger.info("Cleaning events data...")
        
        # Make a copy to avoid modifying the original
        df = events_data.copy()
        
        # Convert column names to lowercase
        df.columns = map(str.lower, df.columns)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed duplicates. {len(events_data) - len(df)} duplicate entries found")
        
        # Drop unnecessary columns
        df = df.drop(['location'], axis=1)
        
        # Remove no contests and draws
        df = df.drop(df[(df['result'] == 'nc') | (df['result'] == 'draw')].index)
        logger.info(f"Removed {len(events_data) - len(df)} NC/Draw fights")
        
        # If processing events with odds data, drop rows where odds are 0 or None
        if has_odds:
            initial_len = len(df)
            df = df.drop(df[(df['fighter_odd'].isin([0, None])) | (df['opponent_odd'].isin([0, None])) | (df['fighter_odd'].isna()) | (df['opponent_odd'].isna())].index)
            logger.info(f"Removed {initial_len - len(df)} fights with zero or missing odds")
        
        # Convert result to binary
        df['result'] = df['result'].map({'win': 1})
        
        # Clean time control data
        df['time_control_fighter'] = df['time_control_fighter'].replace(['--', None, 'None', 'nan', 'NaN', 'NAN', 'na', 'NA', '', ' ', 'null', 'NULL', np.nan], 0)
        df['time_control_opp'] = df['time_control_opp'].replace(['--', None, 'None', 'nan', 'NaN', 'NAN', 'na', 'NA', '', ' ', 'null', 'NULL', np.nan], 0)
        
        # Clean weight class names
        df['weight_class'] = df['weight_class'].str.replace('\'', '')
        df['weight_class'] = df['weight_class'].str.replace(' ', '_')
        
        # Clean method names
        df['method'] = df['method'].str.replace('/', '_')
        df['method'] = df['method'].str.replace('-', '_')
        
        # Drop submethod column
        df = df.drop(['submethod'], axis=1)
        
        # Fill NA values in bonus columns
        bonus_columns = ['belt', 'perf', 'fight', 'ko', 'sub']
        df[bonus_columns] = df[bonus_columns].fillna(0)
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info("Events data cleaning completed")
        return df

    def _get_latest_raw_data(self):
        """Load the most recent raw data files"""
        logger.info("Loading latest raw data...")
        
        # Find latest fighters data
        fighters_path = Path("data/raw/fighters")
        if not fighters_path.exists():
            raise FileNotFoundError("No fighters data directory found")
            
        fighters_dates = [d for d in fighters_path.iterdir() if d.is_dir()]
        if not fighters_dates:
            raise FileNotFoundError("No fighters data found")
            
        latest_fighters_date = max(fighters_dates)
        fighters_data = pd.read_pickle(latest_fighters_date / "fighters_data.pkl")
        logger.info(f"Loaded fighters data from {latest_fighters_date}")
        
        # Find latest events data
        events_path = Path("data/raw/events")
        if not events_path.exists():
            raise FileNotFoundError("No events data directory found")
            
        events_dates = [d for d in events_path.iterdir() if d.is_dir()]
        if not events_dates:
            raise FileNotFoundError("No events data found")
            
        latest_events_date = max(events_dates)
        events_data = pd.read_pickle(latest_events_date / "events_data.pkl")
        logger.info(f"Loaded events data from {latest_events_date}")
        
        # Find latest events with odds data
        events_with_odds_path = Path("data/raw/events_with_odds")
        if not events_with_odds_path.exists():
            raise FileNotFoundError("No events with odds data directory found")
        
        events_with_odds_dates = [d for d in events_with_odds_path.iterdir() if d.is_dir()]
        if not events_with_odds_dates:
            raise FileNotFoundError("No events data found")
 
        latest_events_with_odds_date = max(events_with_odds_dates)
        events_with_odds_data = pd.read_pickle(latest_events_with_odds_date / "events_with_odds_data.pkl")
        logger.info(f"Loaded events with odds data from {latest_events_with_odds_date}")
        
        return fighters_data, events_data, events_with_odds_data

    def clean_data(self):
        """
        Clean and prepare data for feature engineering.
        Always processes the latest raw data and saves the results,
        overwriting any existing cleaned data.
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # Load latest raw data
            fighters_data, events_data, events_with_odds_data = self._get_latest_raw_data()
            
            # Clean the data
            logger.info("Processing raw data...")
            
            # Clean fighters data
            cleaned_fighters = self._clean_fighters_data(fighters_data)
            
            # Clean events data
            cleaned_events = self._clean_events_data(events_data)
            
            # Clean events with odds data
            cleaned_events_with_odds = self._clean_events_data(events_with_odds_data, has_odds=True)
            
            # Save cleaned data
            logger.info(f"Saving cleaned data to {self.output_dir}")
            
            # Save fighters data
            cleaned_fighters.to_pickle(self.output_dir / "cleaned_fighters.pkl")
            cleaned_fighters.to_csv(self.output_dir / "cleaned_fighters.csv", index=False)
            logger.info(f"Saved cleaned fighters data: {len(cleaned_fighters)} rows")
            
            # Save events data
            cleaned_events.to_pickle(self.output_dir / "cleaned_events.pkl")
            cleaned_events.to_csv(self.output_dir / "cleaned_events.csv", index=False)
            logger.info(f"Saved cleaned events data: {len(cleaned_events)} rows")
            
            # Save events with odds data
            cleaned_events_with_odds.to_pickle(self.output_dir / "cleaned_events_with_odds.pkl")
            cleaned_events_with_odds.to_csv(self.output_dir / "cleaned_events_with_odds.csv", index=False)
            logger.info(f"Saved cleaned events with odds data: {len(cleaned_events_with_odds)} rows")
            
            return cleaned_fighters, cleaned_events, cleaned_events_with_odds
            
        except Exception as e:
            logger.error(f"Error during data cleaning: {str(e)}")
            raise

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.clean_data() 