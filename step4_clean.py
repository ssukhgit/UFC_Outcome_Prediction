"""
Clean scraped data for UFC fight prediction model.
This script handles data cleaning and standardization.
"""

import logging
from src.data.data_cleaner import DataCleaner
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def clean_data():
    """
    Clean and standardize the raw data
    """
    logger.info("Starting data cleaning process...")
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_fighters, cleaned_events, cleaned_events_with_odds = cleaner.clean_data()
    
    if cleaned_fighters is None or cleaned_events is None or cleaned_events_with_odds is None:
        logger.error("Failed to get cleaned data")
        return False
        
    logger.info("Data cleaning completed successfully!")
    logger.info(f"Cleaned fighters data shape: {cleaned_fighters.shape}")
    logger.info(f"Cleaned events data shape: {cleaned_events.shape}")
    logger.info(f"Cleaned events with odds data shape: {cleaned_events_with_odds.shape}")
    return True

if __name__ == "__main__":
    success = clean_data()
    if not success:
        exit(1) 