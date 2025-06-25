from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import HTTPError
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default headers and settings for scraping
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer": "https://google.com",
    "DNT": "1",
    "Sec-GPC": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-User": "?1",
    "Priority": "u=0, i",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache"
}

IMPERSONATE = 'chrome110'
# Try without proxy first, if needed we can add a working proxy later
PROXY = None

@dataclass
class FightEvent:
    """Represents a fight event for odds scraping"""
    id: int
    name: str
    date: datetime
    location: str
    fighter: str
    opponent: str
    fighter_odd: Optional[int] = None
    opponent_odd: Optional[int] = None

class OddsScraper:
    """Scrapes fight odds from bestfightodds.com"""
    
    BASE_DOMAIN = 'https://www.bestfightodds.com'
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    def __init__(self) -> None:
        self.session = AsyncSession(
            headers=DEFAULT_HEADERS,
            impersonate=IMPERSONATE,
            verify=False,
            proxy=PROXY
        )
        if PROXY:
            self.session.proxies = {
                "http": PROXY,
                "https": PROXY
            }

    async def _make_request(self, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
        """Make HTTP request with retry logic"""
        for attempt in range(retries):
            try:
                resp = await self.session.get(url, verify=False)
                return resp
            except HTTPError as e:
                if attempt == retries - 1:  # Last attempt
                    logger.error(f"Failed to fetch {url} after {retries} attempts: {str(e)}")
                    return None
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {self.RETRY_DELAY}s...")
                await asyncio.sleep(self.RETRY_DELAY)
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {str(e)}")
                return None

    async def get_fighter_urls(self, fighter_name: str) -> dict[str, int]:
        """
        Returns URLs to fighter profiles sorted by match relevance.
        
        Args:
            fighter_name: Name of the fighter to search for
        
        Returns:
            Dictionary mapping profile URLs to their match relevance scores
        """
        resp = await self._make_request(f'{self.BASE_DOMAIN}/search?query={fighter_name}')
        if not resp:
            return {}
        
        # If fighter is unique, site redirects to their page
        if resp.redirect_count > 0:
            return {resp.url.split(self.BASE_DOMAIN)[1]: 100}
            
        soup = BeautifulSoup(resp.content, 'html.parser')
        content_list = soup.find('table', {'class': 'content-list'})
        if not content_list:
            logger.error(f'Fighter {fighter_name} not found')
            return {}
        
        # Find all search results and calculate match scores
        rows = content_list.find_all('tr')
        results = {}
        for row in rows:
            ratio = fuzz.ratio(row.text, fighter_name)
            url = row.find('a').get('href')
            results[url] = ratio
        
        # Sort by match score descending
        return {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    def _convert_date(self, date: datetime) -> str:
        """Convert datetime to site's date format (e.g., 'Jan 1st 2024')"""
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        def get_suffix(day: int) -> str:
            if 10 <= day % 100 <= 20:
                return "th"
            suffixes = {1: "st", 2: "nd", 3: "rd"}
            return suffixes.get(day % 10, "th")
        
        day = date.day
        month = months[date.month - 1]
        year = date.year
        suffix = get_suffix(day)
        
        return f"{month} {day}{suffix} {year}"

    async def get_odds(self, fighter_url: str, date: datetime) -> Tuple[Optional[int], Optional[int]]:
        """
        Get odds for fighter and opponent from fighter's profile page.
        
        Args:
            fighter_url: URL to fighter's profile
            date: Date of the fight
        
        Returns:
            Tuple of (fighter_odd, opponent_odd) or (None, None) if not found
        """
        resp = await self._make_request(f'{self.BASE_DOMAIN}{fighter_url}')
        if not resp:
            return None, None
            
        soup = BeautifulSoup(resp.content, 'html.parser')
        
        # Try exact date and next day (for timezone differences)
        site_date = self._convert_date(date)
        date_td = soup.find('td', string=site_date)
        if not date_td:
            site_date = self._convert_date(date + timedelta(days=1))
            date_td = soup.find('td', string=site_date)
            if not date_td:
                logger.warning(f'Event with {date} date not found')
                return None, None
        
        # Get odds from the table
        opponent_row = date_td.parent.parent
        fighter_row = opponent_row.previous_sibling
        
        try:
            fighter_odd = int(fighter_row.find('td', {'class': 'moneyline'}).text)
            opponent_odd = int(opponent_row.find('td', {'class': 'moneyline'}).text)
            return fighter_odd, opponent_odd
        except (AttributeError, ValueError) as e:
            logger.error(f'Error parsing odds: {str(e)}')
            return None, None

    async def get_fight_odds(self, event: FightEvent) -> Tuple[Optional[int], Optional[int]]:
        """
        Get odds for a fight by trying both fighter and opponent profiles.
        
        Args:
            event: Fight event details
        
        Returns:
            Tuple of (fighter_odd, opponent_odd) or (None, None) if not found
        """
        # Try finding odds through fighter's profile
        fighter_urls = await self.get_fighter_urls(event.fighter)
        for url in fighter_urls:
            try:
                fighter_odd, opponent_odd = await self.get_odds(url, event.date)
                if fighter_odd is not None:
                    logger.info(f'Found odds for {event.fighter} vs {event.opponent}: {fighter_odd} vs {opponent_odd}')
                    return fighter_odd, opponent_odd
            except Exception as e:
                logger.error(f'Error getting odds through {event.fighter}: {str(e)}')
        
        # If not found, try through opponent's profile
        opponent_urls = await self.get_fighter_urls(event.opponent)
        for url in opponent_urls:
            try:
                opponent_odd, fighter_odd = await self.get_odds(url, event.date)
                if opponent_odd is not None:
                    logger.info(f'Found odds for {event.fighter} vs {event.opponent}: {fighter_odd} vs {opponent_odd}')
                    return fighter_odd, opponent_odd
            except Exception as e:
                logger.error(f'Error getting odds through {event.opponent}: {str(e)}')
        
        return None, None 