import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from datetime import datetime
import logging
from pathlib import Path
import time
from ..utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class EventsScraper:
    def __init__(self, base_url="http://www.ufcstats.com/statistics/events/completed?page=all"):
        self.base_url = base_url
        self.timestamp = datetime.now().strftime("%Y%m%d")
        self.output_dir = Path(f"data/raw/events/{self.timestamp}")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko"
        }
        
    def _get_latest_data_path(self):
        """Get path to the most recent events data if it exists and contains data"""
        base_path = Path('data/raw/events')
        if not base_path.exists():
            return None
            
        # Get all date folders
        date_folders = [d for d in base_path.iterdir() if d.is_dir()]
        if not date_folders:
            return None
            
        # Sort folders by date (newest first)
        date_folders.sort(reverse=True)
        
        # Find first folder that contains valid data
        for folder in date_folders:
            data_file = folder / 'events_data.pkl'
            if data_file.exists() and data_file.stat().st_size > 0:
                try:
                    # Try to load the file to verify it's valid
                    pd.read_pickle(data_file)
                    return data_file
                except:
                    logger.warning(f"Found corrupted data file in {folder}, skipping...")
                    continue
        
        return None
        
    def _load_existing_data(self):
        """Load most recent events data if it exists"""
        latest_path = self._get_latest_data_path()
        if latest_path:
            return pd.read_pickle(latest_path)
        return None

    def _parse_time_control(self, time_str):
        """Parse time control string into seconds"""
        if time_str == "--":
            return None
        try:
            dt = datetime.strptime(time_str, '%M:%S')
            return dt.minute * 60 + dt.second
        except ValueError:
            logger.warning(f"Invalid time format: {time_str}")
            return None

    def _parse_fight_details(self, fight_soup, fighter_name, first_name):
        """Parse detailed fight statistics from fight page"""
        if fight_soup.find_all('section', class_="b-fight-details__section js-fight-section")[0].text.strip() == 'Round-by-round stats not currently available.':
            return None
            
        stat_tag = fight_soup.find_all('section', class_="b-fight-details__section js-fight-section")[1]
        nrow = 2
        fi, op = (0, 1) if first_name == fighter_name else (1, 0)
        
        try:
            stats = {
                'str_total_fighter': int(stat_tag.find_all('p', class_="b-fight-details__table-text")[nrow*2 + fi].text.strip().split()[-1]),
                'td_total_fighter': int(stat_tag.find_all('p', class_="b-fight-details__table-text")[nrow*5 + fi].text.strip().split()[-1]),
                'reversals_fighter': stat_tag.find_all('p', class_="b-fight-details__table-text")[nrow*8 + fi].text.strip(),
                'time_ctrl_fighter': self._parse_time_control(stat_tag.find_all('p', class_="b-fight-details__table-text")[nrow*9 + fi].text.strip()),
                'str_total_opp': int(stat_tag.find_all('p', class_="b-fight-details__table-text")[nrow*2 + op].text.strip().split()[-1]),
                'td_total_opp': int(stat_tag.find_all('p', class_="b-fight-details__table-text")[nrow*5 + op].text.strip().split()[-1]),
                'reversals_opp': stat_tag.find_all('p', class_="b-fight-details__table-text")[nrow*8 + op].text.strip(),
                'time_ctrl_opp': self._parse_time_control(stat_tag.find_all('p', class_="b-fight-details__table-text")[nrow*9 + op].text.strip())
            }
            return stats
        except Exception as e:
            logger.error(f"Error parsing fight details: {str(e)}")
            return None

    def _parse_bonus_counts(self, bonus_images):
        """Parse bonus images into a dictionary of counts"""
        bonus_counts = {}
        if bonus_images:
            for bonus in bonus_images:
                bonus_type = bonus['src'].split('/')[-1].split('.')[0]
                bonus_counts[bonus_type] = bonus_counts.get(bonus_type, 0) + 1
        return bonus_counts

    def scrape_events(self):
        """
        Scrape event data from UFCStats.
        If existing data is found, only scrape new events.
        """
        # Load existing data if available
        existing_data = self._load_existing_data()
        if existing_data is not None:
            logger.info(f"Found existing data with {len(existing_data)} fights")
            existing_events = set(existing_data['Event_Name'].unique())
        else:
            logger.info("No existing data found, will scrape all events")
            existing_events = set()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Fetch the event list page
        response = requests.get(self.base_url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all event links
        events_links = soup.find_all('a', class_='b-link b-link_style_black')
        logger.info(f"Found {len(events_links)} events to process")
        
        fights_data = []
        processed_count = 0
        count = 0

        # Iterate through each event link
        for event_link in events_links:
            try:
                event_url = event_link['href']
                event_response = requests.get(event_url, headers=self.headers)
                event_soup = BeautifulSoup(event_response.text, 'html.parser')
                count += 1
                
                # Get event name, location and date
                event_name = event_soup.find('span', class_='b-content__title-highlight').text.strip()
                
                # Skip if event already exists in data
                if event_name in existing_events:
                    logger.info(f"Skipping existing event: {event_name} {count}/{len(events_links)}")
                    continue
                    
                logger.info(f"Processing event: {event_name} {count}/{len(events_links)}")
                
                date_tag = event_soup.find_all('li', class_='b-list__box-list-item')[0]
                event_date = date_tag.get_text(strip=True)[5:]
                location_tag = event_soup.find_all('li', class_='b-list__box-list-item')[1]
                location = location_tag.get_text(strip=True)[9:]
                
                fight_history = event_soup.find_all('tr', class_='b-fight-details__table-row')
                
                # Process each fight in the event
                for i, fight in enumerate(fight_history):
                    if i == 0:  # Skip header row
                        continue
                        
                    try:
                        columns = fight.find_all('td')
                        if columns[0].text.strip() == 'next':
                            continue
                        
                        # Extract fight details
                        result = columns[0].find('a').text.strip()
                        fighter_name = columns[1].find_all('a', class_='b-link b-link_style_black')[0].text.strip()
                        opp_name = columns[1].find_all('a', class_='b-link b-link_style_black')[1].text.strip()
                        
                        # Basic stats
                        kd_fighter = columns[2].find_all('p', class_='b-fight-details__table-text')[0].text.strip()
                        str_fighter = columns[3].find_all('p', class_='b-fight-details__table-text')[0].text.strip()
                        td_fighter = columns[4].find_all('p', class_='b-fight-details__table-text')[0].text.strip()
                        sub_fighter = columns[5].find_all('p', class_='b-fight-details__table-text')[0].text.strip()
                        
                        kd_opp = columns[2].find_all('p', class_='b-fight-details__table-text')[1].text.strip()
                        str_opp = columns[3].find_all('p', class_='b-fight-details__table-text')[1].text.strip()
                        td_opp = columns[4].find_all('p', class_='b-fight-details__table-text')[1].text.strip()
                        sub_opp = columns[5].find_all('p', class_='b-fight-details__table-text')[1].text.strip()
                        
                        # Get detailed fight stats
                        fight_link = columns[0].find('a')['href']
                        fight_response = requests.get(fight_link, headers=self.headers)
                        fight_soup = BeautifulSoup(fight_response.text, 'html.parser')
                        
                        # Get detailed stats
                        first_name = fight_soup.find_all('section', class_="b-fight-details__section js-fight-section")[1].find_all('p', class_="b-fight-details__table-text")[0].text.strip()
                        detailed_stats = self._parse_fight_details(fight_soup, fighter_name, first_name)
                        
                        if detailed_stats is None:
                            logger.warning(f"Skipping fight due to missing round-by-round stats: {fighter_name} vs {opp_name}")
                            continue
                        
                        # Get weight class and method
                        weight_class = columns[6].find_all('p', class_='b-fight-details__table-text')[0].text.strip()
                        bonus_counts = self._parse_bonus_counts(columns[6].find_all('p', class_='b-fight-details__table-text')[0].find_all('img'))
                        
                        method = columns[7].find_all('p', class_='b-fight-details__table-text')[0].text.strip()
                        submethod = columns[7].find_all('p', class_='b-fight-details__table-text')[1].text.strip()
                        round_ = int(columns[8].text.strip())
                        time_ = columns[9].text.strip()
                        
                        # Create fight record
                        fight_data = {
                            "Event_Name": event_name,
                            "Event_Date": datetime.strptime(event_date, '%B %d, %Y').date(),
                            "Location": location,
                            "Fighter_Name": fighter_name,
                            "Opp_Name": opp_name,
                            "Result": result,
                            "KD_Fighter": kd_fighter,
                            "STR_Fighter": str_fighter,
                            "STR_TOTAL_Fighter": detailed_stats['str_total_fighter'],
                            "TD_Fighter": td_fighter,
                            "TD_TOTAL_Fighter": detailed_stats['td_total_fighter'],
                            "SUB_Fighter": sub_fighter,
                            "REVS_Fighter": detailed_stats['reversals_fighter'],
                            "Time_Control_Fighter": detailed_stats['time_ctrl_fighter'],
                            "KD_Opp": kd_opp,
                            "STR_Opp": str_opp,
                            "STR_TOTAL_Opp": detailed_stats['str_total_opp'],
                            "TD_Opp": td_opp,
                            "TD_TOTAL_Opp": detailed_stats['td_total_opp'],
                            "SUB_Opp": sub_opp,
                            "REVS_Opp": detailed_stats['reversals_opp'],
                            "Time_Control_Opp": detailed_stats['time_ctrl_opp'],
                            "Weight_Class": weight_class,
                            "Method": method,
                            "Submethod": submethod,
                            "Round": round_,
                            "Time": self._parse_time_control(time_)
                        }
                        
                        # Add bonus information
                        fight_data.update(bonus_counts)
                        fights_data.append(fight_data)
                        processed_count += 1
                        
                        time.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        logger.error(f"Error processing fight in event {event_name}: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing event at {event_url if 'event_url' in locals() else 'unknown URL'}: {str(e)}")
                continue
                
            logger.info(f"Processed {processed_count} fights so far")
        
        # If we have new data to save
        if fights_data:
            # Create new DataFrame with scraped data
            new_data = pd.DataFrame(fights_data)
            
            # Remove duplicates
            new_data = new_data.drop_duplicates()
            
            # Combine with existing data if available
            if existing_data is not None:
                final_data = pd.concat([existing_data, new_data], ignore_index=True)
                logger.info(f"Added {len(new_data)} new fights to existing {len(existing_data)} records")
            else:
                final_data = new_data
                logger.info(f"Created new dataset with {len(new_data)} fights")
            
            # Save to timestamped directory
            final_data.to_pickle(self.output_dir / 'events_data.pkl')
            final_data.to_csv(self.output_dir / 'events_data.csv', index=False)
            logger.info(f"Saved data to {self.output_dir}")
            
            return final_data
        else:
            logger.info("No new fights found to add to the dataset")
            return existing_data

if __name__ == "__main__":
    scraper = EventsScraper()
    scraper.scrape_events() 