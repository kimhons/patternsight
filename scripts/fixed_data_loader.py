#!/usr/bin/env python3
"""
Fixed Data Loader for PatternSight v4.0
Loads real lottery data from provided JSON files
"""

import json
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def load_real_lottery_data():
    """Load all available real lottery data"""
    lottery_data = {}
    
    # Data file mappings
    data_files = {
        'powerball': '/home/ubuntu/upload/powerball_data_5years.json',
        'mega_millions': '/home/ubuntu/upload/megamillions.json',
        'lucky_for_life': '/home/ubuntu/upload/luckyforlife.json',
        'cash4life': '/home/ubuntu/upload/cash4life_fixed.json',
        'take5': '/home/ubuntu/upload/take5.json',
        'pick10': '/home/ubuntu/upload/pick10.json'
    }
    
    for lottery_type, file_path in data_files.items():
        try:
            data = load_lottery_file(file_path, lottery_type)
            if not data.empty:
                lottery_data[lottery_type] = data
                logger.info(f"✅ Loaded {len(data)} draws for {lottery_type}")
            else:
                logger.warning(f"⚠️ No data loaded for {lottery_type}")
        except Exception as e:
            logger.error(f"❌ Failed to load {lottery_type}: {e}")
    
    return lottery_data

def load_lottery_file(file_path: str, lottery_type: str) -> pd.DataFrame:
    """Load lottery data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Handle different data formats
        if isinstance(raw_data, dict) and 'error' in raw_data:
            logger.warning(f"Error in data file {file_path}: {raw_data.get('message', 'Unknown error')}")
            return pd.DataFrame()
        
        if not isinstance(raw_data, list):
            logger.error(f"Unexpected data format in {file_path}")
            return pd.DataFrame()
        
        draws = []
        
        for entry in raw_data:
            try:
                if lottery_type == 'powerball':
                    # Powerball format
                    if 'draw_date' in entry and 'winning_numbers' in entry:
                        date_str = entry['draw_date']
                        numbers_str = entry['winning_numbers']
                        powerball = entry.get('powerball', entry.get('mega_ball'))
                        
                        draw_date = datetime.fromisoformat(date_str.replace('T00:00:00.000', ''))
                        numbers = [int(x) for x in numbers_str.split()]
                        
                        draw_entry = {
                            'date': draw_date,
                            'numbers': sorted(numbers),
                            'bonus': int(powerball) if powerball else None,
                            'day_of_week': draw_date.strftime('%A'),
                            'month': draw_date.month,
                            'year': draw_date.year
                        }
                        draws.append(draw_entry)
                
                elif lottery_type == 'mega_millions':
                    # Mega Millions format
                    if 'draw_date' in entry and 'winning_numbers' in entry:
                        date_str = entry['draw_date']
                        numbers_str = entry['winning_numbers']
                        mega_ball = entry.get('mega_ball')
                        
                        draw_date = datetime.fromisoformat(date_str.replace('T00:00:00.000', ''))
                        numbers = [int(x) for x in numbers_str.split()]
                        
                        draw_entry = {
                            'date': draw_date,
                            'numbers': sorted(numbers),
                            'bonus': int(mega_ball) if mega_ball else None,
                            'day_of_week': draw_date.strftime('%A'),
                            'month': draw_date.month,
                            'year': draw_date.year
                        }
                        draws.append(draw_entry)
                
                elif lottery_type == 'lucky_for_life':
                    # Lucky for Life format
                    if 'draw_date' in entry and 'winning_numbers' in entry:
                        date_str = entry['draw_date']
                        numbers_str = entry['winning_numbers']
                        lucky_ball = entry.get('lucky_ball')
                        
                        draw_date = datetime.fromisoformat(date_str.replace('T00:00:00.000', ''))
                        numbers = [int(x) for x in numbers_str.split()]
                        
                        draw_entry = {
                            'date': draw_date,
                            'numbers': sorted(numbers),
                            'bonus': int(lucky_ball) if lucky_ball else None,
                            'day_of_week': draw_date.strftime('%A'),
                            'month': draw_date.month,
                            'year': draw_date.year
                        }
                        draws.append(draw_entry)
                
                elif lottery_type == 'cash4life':
                    # Cash4Life format
                    if 'draw_date' in entry and 'winning_numbers' in entry:
                        date_str = entry['draw_date']
                        numbers_str = entry['winning_numbers']
                        cash_ball = entry.get('cash_ball')
                        
                        draw_date = datetime.fromisoformat(date_str.replace('T00:00:00.000', ''))
                        numbers = [int(x) for x in numbers_str.split()]
                        
                        draw_entry = {
                            'date': draw_date,
                            'numbers': sorted(numbers),
                            'bonus': int(cash_ball) if cash_ball else None,
                            'day_of_week': draw_date.strftime('%A'),
                            'month': draw_date.month,
                            'year': draw_date.year
                        }
                        draws.append(draw_entry)
                
                elif lottery_type in ['take5', 'pick10']:
                    # Take 5 and Pick 10 format (no bonus ball)
                    if 'draw_date' in entry and 'winning_numbers' in entry:
                        date_str = entry['draw_date']
                        numbers_str = entry['winning_numbers']
                        
                        draw_date = datetime.fromisoformat(date_str.replace('T00:00:00.000', ''))
                        numbers = [int(x) for x in numbers_str.split()]
                        
                        draw_entry = {
                            'date': draw_date,
                            'numbers': sorted(numbers),
                            'day_of_week': draw_date.strftime('%A'),
                            'month': draw_date.month,
                            'year': draw_date.year
                        }
                        draws.append(draw_entry)
                
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Skipping invalid entry in {lottery_type}: {e}")
                continue
        
        if draws:
            draws.sort(key=lambda x: x['date'])
            df = pd.DataFrame(draws)
            logger.info(f"Successfully parsed {len(draws)} draws for {lottery_type}")
            return df
        else:
            logger.warning(f"No valid draws found for {lottery_type}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test the data loader
    logging.basicConfig(level=logging.INFO)
    data = load_real_lottery_data()
    
    for lottery_type, df in data.items():
        print(f"\n{lottery_type.upper()}:")
        print(f"  Draws: {len(df)}")
        if not df.empty:
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Sample numbers: {df.iloc[0]['numbers']}")
            if 'bonus' in df.columns and df.iloc[0].get('bonus'):
                print(f"  Sample bonus: {df.iloc[0]['bonus']}")

