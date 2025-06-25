"""
Engineer features for UFC fight prediction model.
This script handles feature creation and transformation.
"""

import logging
import argparse
from src.data.feature_engineer import FeatureEngineer
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def engineer_features(use_odds=False):
    """
    Create and transform features for model training
    
    Args:
        use_odds (bool): Whether to use events with odds data instead of regular events data
    """
    logger.info("Starting feature engineering process...")
    
    # Engineer features
    engineer = FeatureEngineer(use_odds=use_odds)
    features_data = engineer.engineer_features()
    
    if features_data is not None:
        logger.info("Feature engineering completed successfully!")
        logger.info(f"Created features for {len(features_data)} fight records")
        if use_odds:
            logger.info("Used events with odds data for feature engineering")
        else:
            logger.info("Used regular events data for feature engineering")
    else:
        logger.error("Failed to engineer features")
        return False
        
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Engineer features for UFC fight prediction')
    parser.add_argument('--use-odds', action='store_true', help='Use events with odds data instead of regular events data')
    args = parser.parse_args()
    
    engineer_features(use_odds=args.use_odds) 