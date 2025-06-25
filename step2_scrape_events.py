"""
Scrape event data from UFCStats.
This script can be run independently to update event data.
"""

import logging
from src.data.events_scraper import EventsScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scrape_events():
    """
    Scrape and save event data
    """
    logger.info("Scraping events data...")
    events_scraper = EventsScraper()
    events_scraper.scrape_events()
    logger.info("Event data scraping completed!")

if __name__ == "__main__":
    scrape_events() 