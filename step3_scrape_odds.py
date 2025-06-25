import os
import pandas as pd
import asyncio
from datetime import datetime
import logging
from pathlib import Path
from src.data.odds_scraper import OddsScraper, FightEvent
from src.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

async def main():
    """
    Load latest raw events data, scrape odds for each fight, and save updated data.
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Find latest raw events data
    raw_events_dir = Path("data/raw/events")
    latest_date = max(os.listdir(raw_events_dir))
    latest_events_file = raw_events_dir / latest_date / "events_data.pkl"
    
    # Create output directory
    output_dir = Path(f"data/raw/events_with_odds/{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load events data from pickle
    logger.info(f"Loading events data from {latest_events_file}")
    events_df = pd.read_pickle(latest_events_file)
    total_fights = len(events_df)
    logger.info(f"Found {total_fights} fights to process")
    
    # Initialize scraper
    scraper = OddsScraper()
    
    # Process each fight
    for i in range(total_fights):
        row = events_df.iloc[i]
        logger.info(f"[{i + 1}/{total_fights}] Processing: {str(row['Fighter_Name'])} vs {str(row['Opp_Name'])}")
        
        # Convert date string to datetime
        event_date = datetime.strptime(str(row['Event_Date']), '%Y-%m-%d')
        
        # Create event object
        event = FightEvent(
            id=i,
            name=str(row['Event_Name']),
            date=event_date,
            location=str(row['Location']),
            fighter=str(row['Fighter_Name']),
            opponent=str(row['Opp_Name'])
        )
        
        # Get odds
        fighter_odd, opponent_odd = await scraper.get_fight_odds(event)
        
        # Update DataFrame
        events_df.at[i, 'Fighter_Odd'] = fighter_odd
        events_df.at[i, 'Opponent_Odd'] = opponent_odd
    
    # Save results in both formats
    output_csv = output_dir / "events_with_odds_data.csv"
    output_pkl = output_dir / "events_with_odds_data.pkl"
    
    events_df.to_csv(output_csv, index=False)
    events_df.to_pickle(output_pkl)
    
    logger.info(f"Saved events with odds to:")
    logger.info(f"- CSV: {output_csv}")
    logger.info(f"- Pickle: {output_pkl}")

if __name__ == "__main__":
    asyncio.run(main()) 