"""
Scrape fighter data from UFCStats.
This script can be run independently to update fighter data.
"""

import logging
from src.data.fighters_scraper import FightersScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_fighters():
    """
    Scrape and save fighter data
    """
    logger.info("Scraping fighters data...")
    fighters_scraper = FightersScraper()
    fighters_scraper.scrape_fighters()
    logger.info("Fighter data scraping completed!")

if __name__ == "__main__":
    scrape_fighters()