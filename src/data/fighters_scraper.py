import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import time
import logging
from ..utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class FightersScraper:
    def __init__(self, base_url="http://www.ufcstats.com/statistics/fighters"):
        self.base_url = base_url
        self.timestamp = datetime.now().strftime("%Y%m%d")
        self.output_dir = Path(f"data/raw/fighters/{self.timestamp}")
        self.headers = {
            "User-Agent": "Mozilla/4.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15"
        }
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        
    def _get_latest_data_path(self):
        """Get path to the most recent fighters data if it exists and contains data"""
        base_path = Path('data/raw/fighters')
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
            data_file = folder / 'fighters_data.pkl'
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
        """Load most recent fighters data if it exists"""
        latest_path = self._get_latest_data_path()
        if latest_path:
            return pd.read_pickle(latest_path)
        return None
        
    def _get_fighter_citizenship(self, fighter_name):
        """Look up fighter's country of citizenship"""
        url = 'https://www.wikidata.org/w/api.php'
        params = {
            'action': 'wbsearchentities',
            'search': fighter_name,
            'language': 'en',
            'format': 'json',
            'limit': 1
        }

        response = requests.get(url, params=params)
        data = response.json()
        if data['search']:
            entity_id = data['search'][0]['id']

            params = {
                'action': 'wbgetclaims',
                'entity': entity_id,
                'property': 'P27',  # Property ID for 'country of citizenship'
                'format': 'json'
            }

            response = requests.get(url, params=params)
            claims = response.json().get('claims', {})
            if 'P27' in claims:
                citizenship_id = claims['P27'][0]['mainsnak']['datavalue']['value']['id']

                params = {
                    'action': 'wbgetentities',
                    'ids': citizenship_id,
                    'languages': 'en',
                    'format': 'json'
                }
                response = requests.get(url, params=params)
                citizenship_data = response.json()
                country_name = citizenship_data['entities'][citizenship_id]['labels']['en']['value']
                return country_name

        return "Not_Found"

    def scrape_fighters(self):
        """
        Scrape fighter data from UFCStats.
        If existing data is found, only scrape new fighters that aren't in the existing dataset.
        """
        # Load existing data if available
        existing_data = self._load_existing_data()
        if existing_data is not None:
            logger.info(f"Found existing data with {len(existing_data)} fighters")
            existing_names = set(existing_data['fighter_name'])
        else:
            logger.info("No existing data found, will scrape all fighters")
            existing_names = set()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        fighters_data = []
        
        for letter in self.letters:
            # Fetch the fighters list page for the given letter
            url = f"{self.base_url}?char={letter}&page=all"
            logger.info(f"Processing fighters starting with letter: {letter}")
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all fighter links
            fighter_links = soup.find_all('a', class_='b-link b-link_style_black')
            
            if not fighter_links:
                logger.warning(f"No fighters found for letter {letter}")
                continue

            # Iterate through each fighter link
            # Track processed URLs to avoid duplicate scraping
            processed_urls = set()
            
            for gfdgfd in fighter_links:
                try:
                    fighter_url = gfdgfd['href']
                    
                    # Skip if we've already processed this URL
                    if fighter_url in processed_urls:
                        logger.debug(f"Skipping already processed URL: {fighter_url}")
                        continue
                    
                    processed_urls.add(fighter_url)
                    fighter_response = requests.get(fighter_url, headers=self.headers)
                    fighter_soup = BeautifulSoup(fighter_response.text, 'html.parser')

                    # Get Fighter's data
                    fighter_name = fighter_soup.find('span', class_='b-content__title-highlight').text.strip()
                    logger.info(f"Found fighter name: {fighter_name}")
                    
                    # Skip if fighter already exists in data
                    if fighter_name in existing_names:
                        logger.debug(f"Skipping existing fighter: {fighter_name}")
                        continue
                        
                    # # Also skip if we just added this fighter (handles duplicates in source)
                    # if fighters_data and fighter_name == fighters_data[-1]['fighter_name']:
                    #     logger.debug(f"Skipping duplicate fighter: {fighter_name}")
                    #     continue

                    logger.info(f"Processing fighter: {fighter_name} ({letter.upper()})")

                    # Extract fighter details
                    detail_items = fighter_soup.find_all('li', class_='b-list__box-list-item b-list__box-list-item_type_block')
                    if len(detail_items) < 5:
                        logger.warning(f"Incomplete fighter details for {fighter_name}, skipping")
                        continue

                    height = detail_items[0].get_text(strip=True)[7:]
                    weight = detail_items[1].get_text(strip=True)[7:]
                    reach = detail_items[2].get_text(strip=True)[6:]
                    stance = detail_items[3].get_text(strip=True)[7:]
                    dob = detail_items[4].get_text(strip=True)[4:]

                    if dob != "--":
                        try:
                            dob = datetime.strptime(dob, '%b %d, %Y').date()
                        except ValueError:
                            logger.warning(f"Invalid date format for {fighter_name}: {dob}")
                            dob = None
                    else:
                        dob = None
                    
                    # Determine fighter's sex from fight history
                    sex = None
                    fight_history = fighter_soup.find_all('tr', class_='b-fight-details__table-row')
                    for i, fight in enumerate(fight_history):
                        if i == 0:  # Skip header row
                            continue
                        try:
                            columns = fight.find_all('td')
                            if not columns or columns[0].text.strip() == 'next':
                                continue
                            
                            fight_link = columns[0].find('a')
                            if not fight_link:
                                continue
                                
                            fight_response = requests.get(fight_link['href'], headers=self.headers)
                            fight_soup = BeautifulSoup(fight_response.text, 'html.parser')
                            
                            fight_title = fight_soup.find('i', class_="b-fight-details__fight-title").text.strip()
                            sex = 'Female' if fight_title[:5] == 'Women' else 'Male'
                            break
                        except Exception as e:
                            logger.warning(f"Error processing fight history for {fighter_name}: {str(e)}")
                            continue

                    # Handle special case for Yizha, because his true country of citizenship is China, but Wiki gives Israel
                    if fighter_name == "Yizha":
                        country = "China"
                    else:
                        country = self._get_fighter_citizenship(fighter_name)
                    
                    fighters_data.append({
                        "fighter_name": fighter_name,
                        "height": height,
                        "weight": weight,
                        "reach": reach,
                        "stance": stance,
                        "sex": sex,
                        "dob": dob,
                        "country": country
                    })

                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error processing fighter at {fighter_url if 'fighter_url' in locals() else 'unknown URL'}: {str(e)}")
                    continue

        # If we have new data to save
        if fighters_data:
            # Create new DataFrame with scraped data
            new_data = pd.DataFrame(fighters_data)
            
            # Combine with existing data if available
            if existing_data is not None:
                final_data = pd.concat([existing_data, new_data], ignore_index=True)
                logger.info(f"Added {len(new_data)} new fighters to existing {len(existing_data)} records")
            else:
                final_data = new_data
                logger.info(f"Created new dataset with {len(new_data)} fighters")
                
            # Save to timestamped directory
            final_data.to_pickle(self.output_dir / 'fighters_data.pkl')
            final_data.to_csv(self.output_dir / 'fighters_data.csv', index=False)
            logger.info(f"Saved data to {self.output_dir}")
            
            return final_data
        else:
            logger.info("No new fighters found to add to the dataset")
            return existing_data

if __name__ == "__main__":
    scraper = FightersScraper()
    scraper.scrape_fighters()