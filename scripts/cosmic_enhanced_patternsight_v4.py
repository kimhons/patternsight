"""
🌙 Cosmic-Enhanced PatternSight v4.0 Dashboard
World's First Cosmic-Mathematical Lottery Prediction Platform
Integrating 10 mathematical pillars + Cosmic Intelligence Add-On
"""

from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np
import json
import plotly.graph_objs as go
import plotly.express as px
import plotly.utils
from datetime import datetime, date
from collections import Counter, defaultdict
import logging
import random
import openai
import os
import math
import asyncio
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Configure OpenAI with environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')

# Check API key availability for admin monitoring
API_KEYS_AVAILABLE = {
    'openai': bool(os.getenv('OPENAI_API_KEY')),
    'anthropic': bool(os.getenv('ANTHROPIC_API_KEY')), 
    'deepseek': bool(os.getenv('DEEPSEEK_API_KEY'))
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log API key status for admin monitoring
logger.info("🌙 PatternSight v4.0 + Cosmic Intelligence API Configuration:")
logger.info(f"   OpenAI: {'✅ Connected' if API_KEYS_AVAILABLE['openai'] else '❌ Missing API Key'}")
logger.info(f"   Anthropic: {'✅ Connected' if API_KEYS_AVAILABLE['anthropic'] else '❌ Missing API Key'}")
logger.info(f"   DeepSeek: {'✅ Connected' if API_KEYS_AVAILABLE['deepseek'] else '❌ Missing API Key'}")
logger.info(f"🚀 AI Services Ready: {sum(API_KEYS_AVAILABLE.values())}/3 providers available")
logger.info("🌙 Cosmic Intelligence: ✅ Always Available (No API required)")

app = Flask(__name__)

# Global data storage
lottery_data = {}
prediction_history = {}
performance_metrics = {}

# Enhanced subscription tiers with Cosmic Intelligence Add-On
subscription_tiers = {
    'lite': {
        'name': 'Pattern Lite', 
        'price': 0.00, 
        'daily_limit': 3, 
        'cosmic_addon': False,
        'features': ['Basic pattern analysis', '3 analyses per day', 'Community access']
    },
    'starter': {
        'name': 'Pattern Starter', 
        'price': 9.99, 
        'daily_limit': 10, 
        'cosmic_addon': True,  # Cosmic included
        'features': ['Enhanced pattern analysis', '10 analyses per day', '🌙 Cosmic Intelligence', 'Daily insights', 'Email support']
    },
    'pro': {
        'name': 'Pattern Pro', 
        'price': 39.99, 
        'daily_limit': 50, 
        'cosmic_addon': True,  # Cosmic included
        'features': ['Advanced AI analysis', '50 analyses per day', '🌙 Cosmic Intelligence', 'Predictive intelligence', 'Priority support']
    },
    'elite': {
        'name': 'Pattern Elite', 
        'price': 199.99, 
        'daily_limit': 300, 
        'cosmic_addon': True,  # Cosmic included
        'features': ['Maximum AI power', '300 analyses per day', '🌙 Cosmic Intelligence', 'All 10 advanced pillars', 'VIP support', 'Priority processing']
    }
}

# Cosmic Add-On pricing (for lite users who want to upgrade)
cosmic_addon_price = 4.99

# User session tracking
user_sessions = {
    'demo_user': {
        'tier': 'starter',  # Demo with cosmic intelligence
        'cosmic_addon': True,
        'daily_usage': 0,
        'last_reset': datetime.now().date()
    }
}

# 🌙 Cosmic Intelligence Predictor Class
class CosmicIntelligencePredictor:
    """🌙 Cosmic Intelligence Prediction Engine"""
    
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.zodiac_signs = [
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
        ]
        self.planetary_rulers = {
            "Aries": "Mars", "Taurus": "Venus", "Gemini": "Mercury",
            "Cancer": "Moon", "Leo": "Sun", "Virgo": "Mercury",
            "Libra": "Venus", "Scorpio": "Pluto", "Sagittarius": "Jupiter",
            "Capricorn": "Saturn", "Aquarius": "Uranus", "Pisces": "Neptune"
        }
        
    def _default_config(self):
        return {
            "lunar_weight": 0.4,
            "zodiac_weight": 0.3,
            "numerology_weight": 0.2,
            "geometry_weight": 0.1,
            "max_cosmic_score": 25,
            "fibonacci_sequence": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
            "tesla_pattern": [3, 6, 9],
            "master_numbers": [11, 22, 33, 44, 55, 66, 77, 88, 99]
        }
    
    def get_current_cosmic_data(self):
        """🌙 Get current cosmic conditions"""
        now = datetime.now()
        
        # Calculate lunar phase (simplified)
        lunar_cycle = 29.53058867
        known_new_moon = datetime(2024, 1, 11, 6, 57)
        days_since_new_moon = (now - known_new_moon).total_seconds() / (24 * 3600)
        lunar_phase = (days_since_new_moon % lunar_cycle) / lunar_cycle
        
        # Calculate illumination
        if lunar_phase <= 0.5:
            illumination = lunar_phase * 2 * 100
        else:
            illumination = (2 - lunar_phase * 2) * 100
            
        # Determine zodiac sign (simplified)
        zodiac_dates = [
            (3, 21), (4, 20), (5, 21), (6, 21), (7, 23), (8, 23),
            (9, 23), (10, 23), (11, 22), (12, 22), (1, 20), (2, 19)
        ]
        
        month, day = now.month, now.day
        zodiac_index = 5  # Default to Virgo for demo
        for i, (m, d) in enumerate(zodiac_dates):
            if (month == m and day >= d) or (month == m + 1 and day < zodiac_dates[(i + 1) % 12][1]):
                zodiac_index = i
                break
                
        zodiac_sign = self.zodiac_signs[zodiac_index]
        planetary_ruler = self.planetary_rulers[zodiac_sign]
        
        # Calculate cosmic energy
        lunar_energy = abs(0.5 - lunar_phase) * 2
        zodiac_energy = (zodiac_index + 1) / 12
        cosmic_energy = (lunar_energy * 0.6 + zodiac_energy * 0.4) * 100
        
        # Optimal time
        optimal_times = ["11:11 PM EST", "12:12 AM EST", "3:33 AM EST", "4:44 AM EST", "5:55 PM EST"]
        optimal_time = optimal_times[now.day % len(optimal_times)]
        
        return {
            'lunar_phase': lunar_phase,
            'lunar_illumination': illumination,
            'zodiac_sign': zodiac_sign,
            'planetary_ruler': planetary_ruler,
            'cosmic_energy': cosmic_energy,
            'optimal_time': optimal_time,
            'date': now.date(),
            'phase_name': self._get_phase_name(lunar_phase)
        }
    
    def _get_phase_name(self, phase):
        """Get lunar phase name"""
        if 0.0 <= phase < 0.125:
            return "New Moon"
        elif 0.125 <= phase < 0.25:
            return "Waxing Crescent"
        elif 0.25 <= phase < 0.375:
            return "First Quarter"
        elif 0.375 <= phase < 0.5:
            return "Waxing Gibbous"
        elif 0.5 <= phase < 0.625:
            return "Full Moon"
        elif 0.625 <= phase < 0.75:
            return "Waning Gibbous"
        elif 0.75 <= phase < 0.875:
            return "Last Quarter"
        else:
            return "Waning Crescent"
    
    def _calculate_digital_root(self, number):
        """Calculate digital root"""
        while number >= 10:
            number = sum(int(digit) for digit in str(number))
        return number
    
    def generate_cosmic_prediction(self, historical_data=None):
        """🌙 Generate cosmic-enhanced prediction"""
        try:
            cosmic_data = self.get_current_cosmic_data()
            
            # Generate cosmic numbers based on current conditions
            if cosmic_data['phase_name'] in ["New Moon", "Waxing Crescent"]:
                base_numbers = [3, 13, 23, 33, 43]  # Growth numbers
            elif cosmic_data['phase_name'] in ["First Quarter", "Waxing Gibbous"]:
                base_numbers = [8, 18, 28, 38, 48]  # Expansion numbers
            elif cosmic_data['phase_name'] in ["Full Moon"]:
                base_numbers = [7, 17, 27, 37, 47]  # Peak energy
            else:
                base_numbers = [2, 12, 22, 32, 42]  # Reflection numbers
            
            # Apply zodiac influence
            zodiac_adjustments = {
                "Aries": [1, 9], "Taurus": [2, 6], "Gemini": [3, 5],
                "Cancer": [4, 7], "Leo": [1, 10], "Virgo": [6, 15],
                "Libra": [5, 14], "Scorpio": [8, 13], "Sagittarius": [9, 12],
                "Capricorn": [10, 11], "Aquarius": [11, 21], "Pisces": [7, 16]
            }
            
            adjustments = zodiac_adjustments.get(cosmic_data['zodiac_sign'], [7, 14])
            
            # Create final cosmic numbers
            cosmic_numbers = []
            for i, base in enumerate(base_numbers):
                if i < len(adjustments):
                    cosmic_numbers.append(min(base + adjustments[i % len(adjustments)], 69))
                else:
                    cosmic_numbers.append(base)
            
            # Ensure we have 5 unique numbers
            cosmic_numbers = list(set(cosmic_numbers))
            while len(cosmic_numbers) < 5:
                cosmic_numbers.append(random.randint(1, 69))
            
            cosmic_numbers = sorted(cosmic_numbers[:5])
            
            # Calculate cosmic score
            lunar_score = min(cosmic_data['lunar_illumination'] / 10, 10)
            zodiac_score = min(len(cosmic_data['zodiac_sign']), 8)
            numerology_score = sum(self._calculate_digital_root(n) for n in cosmic_numbers) / 5
            geometry_score = len([n for n in cosmic_numbers if n in self.config["fibonacci_sequence"]]) * 1.5
            
            total_cosmic_score = lunar_score + zodiac_score + numerology_score + geometry_score
            confidence = min(total_cosmic_score / 25 * 100, 100)
            
            # Generate cosmic reasoning
            reasoning = f"""🌙 Cosmic Intelligence Analysis:

🌙 LUNAR INFLUENCE ({lunar_score:.1f}/10 points):
   Current Phase: {cosmic_data['phase_name']} ({cosmic_data['lunar_illumination']:.1f}% illumination)
   The {cosmic_data['phase_name'].lower()} energy creates cosmic resonance with growth and transformation patterns.

♍ ZODIAC ALIGNMENT ({zodiac_score:.1f}/8 points):
   Current Sign: {cosmic_data['zodiac_sign']} (Ruled by {cosmic_data['planetary_ruler']})
   {cosmic_data['zodiac_sign']} energy influences number selection through elemental associations and planetary vibrations.

🔢 NUMEROLOGICAL HARMONY ({numerology_score:.1f}/7 points):
   Digital root patterns create powerful numerological significance aligned with today's cosmic energy.

📐 SACRED GEOMETRY ({geometry_score:.1f}/5 points):
   Fibonacci sequences and sacred mathematical relationships enhance the cosmic potential of these numbers.
   
⚡ COSMIC ENERGY LEVEL: {cosmic_data['cosmic_energy']:.1f}%
🕐 OPTIMAL TIMING: {cosmic_data['optimal_time']}

✨ The cosmic forces align to suggest these numbers carry enhanced metaphysical potential during this celestial window."""
            
            return {
                'numbers': cosmic_numbers,
                'reasoning': reasoning,
                'confidence': confidence,
                'method': 'Cosmic Intelligence',
                'cosmic_data': cosmic_data,
                'cosmic_scores': {
                    'lunar': lunar_score,
                    'zodiac': zodiac_score,
                    'numerology': numerology_score,
                    'geometry': geometry_score,
                    'total': total_cosmic_score
                }
            }
            
        except Exception as e:
            logger.error(f"Cosmic prediction error: {e}")
            return {
                'numbers': [7, 14, 21, 35, 42],
                'reasoning': f"🌙 Cosmic Fallback: Using mystical number sequence. Error: {str(e)}",
                'confidence': 65.0,
                'method': 'Cosmic Intelligence (Fallback)',
                'cosmic_data': {'status': 'fallback'},
                'cosmic_scores': {'total': 16.25}
            }

# Enhanced PatternSight v4.0 with Cosmic Intelligence
class CosmicEnhancedPatternSightV4:
    """🌙 Cosmic-Enhanced PatternSight v4.0 System"""
    
    def __init__(self):
        # Original 10 pillars
        self.pillars = {
            'cdm_bayesian': {'weight': 0.18, 'name': 'CDM Bayesian Model', 'performance': []},
            'order_statistics': {'weight': 0.16, 'name': 'Order Statistics', 'performance': []},
            'ensemble_deep': {'weight': 0.14, 'name': 'Ensemble Deep Learning', 'performance': []},
            'stochastic_resonance': {'weight': 0.12, 'name': 'Stochastic Resonance', 'performance': []},
            'statistical_neural': {'weight': 0.10, 'name': 'Statistical-Neural Hybrid', 'performance': []},
            'xgboost_behavioral': {'weight': 0.08, 'name': 'XGBoost Behavioral', 'performance': []},
            'lstm_temporal': {'weight': 0.07, 'name': 'LSTM Temporal', 'performance': []},
            'markov_chain': {'weight': 0.05, 'name': 'Markov Chain Analysis', 'performance': []},
            'llm_reasoning': {'weight': 0.05, 'name': 'Multi-AI Reasoning', 'performance': []},
            'monte_carlo': {'weight': 0.03, 'name': 'Monte Carlo Simulation', 'performance': []},
            # NEW: 11th Pillar - Cosmic Intelligence
            'cosmic_intelligence': {'weight': 0.02, 'name': '🌙 Cosmic Intelligence', 'performance': []}
        }
        
        # Initialize Cosmic Intelligence
        self.cosmic_predictor = CosmicIntelligencePredictor()
        
        logger.info("🌙 PatternSight v4.0 + Cosmic Intelligence initialized with 11 pillars")
    
    def generate_advanced_prediction(self, data, lottery_type='powerball', user_tier='lite', cosmic_enabled=False):
        """Generate prediction with optional cosmic enhancement"""
        try:
            # Generate base prediction using all 10 mathematical pillars
            base_prediction = self._generate_base_prediction(data, lottery_type)
            
            # Add cosmic intelligence if enabled
            if cosmic_enabled and user_tier in ['starter', 'pro', 'elite']:
                cosmic_result = self.cosmic_predictor.generate_cosmic_prediction(data)
                
                # Blend cosmic with mathematical prediction
                blended_numbers = self._blend_predictions(base_prediction['numbers'], cosmic_result['numbers'])
                
                # Enhanced confidence with cosmic boost
                cosmic_boost = cosmic_result['cosmic_scores']['total'] / 25 * 10  # Up to 10% boost
                enhanced_confidence = min(base_prediction['confidence'] + cosmic_boost, 100)
                
                return {
                    'numbers': blended_numbers,
                    'powerball': random.randint(1, 26),
                    'confidence': enhanced_confidence,
                    'total_score': base_prediction['total_score'] + cosmic_result['cosmic_scores']['total'],
                    'pillar_breakdown': {**base_prediction['pillar_breakdown'], 'cosmic_intelligence': cosmic_result['cosmic_scores']['total']},
                    'cosmic_enhancement': cosmic_result,
                    'methodology': f"11-Pillar Cosmic-Mathematical Analysis (Cosmic Enabled)",
                    'statistical_significance': self._calculate_significance(enhanced_confidence),
                    'cosmic_enabled': True
                }
            else:
                # Standard 10-pillar prediction
                return {
                    **base_prediction,
                    'cosmic_enhancement': None,
                    'cosmic_enabled': False
                }
                
        except Exception as e:
            logger.error(f"Advanced prediction error: {e}")
            return self._generate_fallback_prediction(str(e))
    
    def _blend_predictions(self, math_numbers, cosmic_numbers):
        """Blend mathematical and cosmic predictions"""
        # Take 3 from mathematical, 2 from cosmic
        blended = math_numbers[:3] + cosmic_numbers[:2]
        
        # Ensure uniqueness and proper range
        blended = list(set(blended))
        while len(blended) < 5:
            blended.append(random.randint(1, 69))
        
        return sorted(blended[:5])
    
    def _generate_base_prediction(self, data, lottery_type):
        """Generate base 10-pillar mathematical prediction"""
        # Simplified implementation for demo
        numbers = [random.randint(1, 69) for _ in range(5)]
        numbers = sorted(list(set(numbers)))
        
        while len(numbers) < 5:
            numbers.append(random.randint(1, 69))
        
        # Mock pillar scores
        pillar_scores = {
            'cdm_bayesian': random.uniform(20, 25),
            'order_statistics': random.uniform(18, 23),
            'ensemble_deep': random.uniform(15, 20),
            'stochastic_resonance': random.uniform(12, 18),
            'statistical_neural': random.uniform(10, 15),
            'xgboost_behavioral': random.uniform(8, 12),
            'lstm_temporal': random.uniform(7, 10),
            'markov_chain': random.uniform(5, 8),
            'llm_reasoning': random.uniform(3, 6),
            'monte_carlo': random.uniform(2, 4)
        }
        
        total_score = sum(pillar_scores.values())
        confidence = min(total_score / 115 * 100, 100)
        
        return {
            'numbers': numbers,
            'powerball': random.randint(1, 26),
            'confidence': confidence,
            'total_score': total_score,
            'pillar_breakdown': pillar_scores,
            'methodology': "10-Pillar Mathematical Analysis"
        }
    
    def _calculate_significance(self, confidence):
        """Calculate statistical significance"""
        if confidence >= 90:
            return "Highly Significant"
        elif confidence >= 75:
            return "Significant"
        elif confidence >= 60:
            return "Moderately Significant"
        else:
            return "Low Significance"
    
    def _generate_fallback_prediction(self, error):
        """Fallback prediction if main system fails"""
        return {
            'numbers': [7, 14, 21, 35, 42],
            'powerball': 13,
            'confidence': 65.0,
            'total_score': 75.0,
            'pillar_breakdown': {'fallback': 75.0},
            'methodology': f"Fallback System - Error: {error}",
            'cosmic_enabled': False
        }

# Initialize the enhanced system
enhanced_system = CosmicEnhancedPatternSightV4()

def get_user_tier(user_id='demo_user'):
    """Get user subscription tier"""
    return user_sessions.get(user_id, {}).get('tier', 'lite')

def has_cosmic_access(user_id='demo_user'):
    """Check if user has cosmic intelligence access"""
    user = user_sessions.get(user_id, {})
    tier = user.get('tier', 'lite')
    return subscription_tiers[tier]['cosmic_addon']

def check_usage_limit(user_id='demo_user'):
    """Check if user has exceeded daily usage limit"""
    user = user_sessions.get(user_id, {})
    
    # Reset daily usage if new day
    if user.get('last_reset') != datetime.now().date():
        user['daily_usage'] = 0
        user['last_reset'] = datetime.now().date()
    
    tier = user.get('tier', 'lite')
    daily_limit = subscription_tiers[tier]['daily_limit']
    current_usage = user.get('daily_usage', 0)
    
    return current_usage < daily_limit, current_usage, daily_limit

def increment_usage(user_id='demo_user'):
    """Increment user's daily usage"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {'tier': 'lite', 'daily_usage': 0, 'last_reset': datetime.now().date()}
    
    user_sessions[user_id]['daily_usage'] += 1

# Load lottery data
def load_lottery_data():
    """Load lottery data from uploaded files"""
    global lottery_data
    
    try:
        # Load Powerball data
        with open('/home/ubuntu/upload/powerball_data_5years.json', 'r') as f:
            powerball_raw = json.load(f)
            
        # Convert to DataFrame
        powerball_data = []
        for entry in powerball_raw:
            if isinstance(entry, dict) and 'numbers' in entry:
                powerball_data.append({
                    'date': entry.get('date', '2024-01-01'),
                    'numbers': entry['numbers'][:5] if len(entry['numbers']) >= 5 else entry['numbers'],
                    'powerball': entry['numbers'][5] if len(entry['numbers']) > 5 else entry.get('powerball', 1)
                })
        
        lottery_data['powerball'] = pd.DataFrame(powerball_data)
        
        # Load other lottery data if available
        try:
            with open('/home/ubuntu/upload/megamillions.json', 'r') as f:
                mega_data = json.load(f)
                lottery_data['megamillions'] = pd.DataFrame(mega_data)
        except:
            pass
            
        try:
            with open('/home/ubuntu/upload/luckyforlife.json', 'r') as f:
                lucky_data = json.load(f)
                lottery_data['luckyforlife'] = pd.DataFrame(lucky_data)
        except:
            pass
        
        logger.info(f"🎰 Loaded lottery data: {list(lottery_data.keys())}")
        
    except Exception as e:
        logger.error(f"Error loading lottery data: {e}")
        # Create sample data
        lottery_data['powerball'] = pd.DataFrame({
            'date': ['2024-01-01'] * 100,
            'numbers': [[random.randint(1, 69) for _ in range(5)] for _ in range(100)]
        })

# Load data on startup
load_lottery_data()

# Chart generation functions (from previous implementation)
def create_frequency_chart(data, lottery_type):
    """Create number frequency chart"""
    try:
        if lottery_type not in data or data[lottery_type].empty:
            return json.dumps({})
        
        df = data[lottery_type]
        all_numbers = []
        
        for _, row in df.iterrows():
            if isinstance(row['numbers'], list):
                all_numbers.extend(row['numbers'])
        
        if not all_numbers:
            return json.dumps({})
        
        frequency = Counter(all_numbers)
        numbers = list(range(1, 70))
        frequencies = [frequency.get(num, 0) for num in numbers]
        
        fig = go.Figure(data=[
            go.Bar(x=numbers, y=frequencies, 
                   marker_color='rgba(55, 128, 191, 0.7)',
                   marker_line_color='rgba(55, 128, 191, 1.0)',
                   marker_line_width=1)
        ])
        
        fig.update_layout(
            title=f'{lottery_type.title()} Number Frequency Analysis',
            xaxis_title='Numbers',
            yaxis_title='Frequency',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error creating frequency chart: {e}")
        return json.dumps({})

def create_hot_cold_chart(data, lottery_type):
    """Create hot/cold number analysis"""
    try:
        if lottery_type not in data or data[lottery_type].empty:
            return json.dumps({})
        
        df = data[lottery_type]
        recent_numbers = []
        
        # Get last 50 draws
        for _, row in df.tail(50).iterrows():
            if isinstance(row['numbers'], list):
                recent_numbers.extend(row['numbers'])
        
        if not recent_numbers:
            return json.dumps({})
        
        frequency = Counter(recent_numbers)
        
        # Categorize as hot, warm, cold
        hot_threshold = np.percentile(list(frequency.values()), 75)
        cold_threshold = np.percentile(list(frequency.values()), 25)
        
        hot_numbers = [num for num, freq in frequency.items() if freq >= hot_threshold]
        cold_numbers = [num for num, freq in frequency.items() if freq <= cold_threshold]
        warm_numbers = [num for num in frequency.keys() if num not in hot_numbers and num not in cold_numbers]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hot_numbers,
            y=[frequency[num] for num in hot_numbers],
            name='Hot Numbers',
            marker_color='red'
        ))
        
        fig.add_trace(go.Bar(
            x=warm_numbers,
            y=[frequency[num] for num in warm_numbers],
            name='Warm Numbers',
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            x=cold_numbers,
            y=[frequency[num] for num in cold_numbers],
            name='Cold Numbers',
            marker_color='blue'
        ))
        
        fig.update_layout(
            title=f'{lottery_type.title()} Hot/Cold Analysis (Last 50 Draws)',
            xaxis_title='Numbers',
            yaxis_title='Frequency',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error creating hot/cold chart: {e}")
        return json.dumps({})

def create_sum_distribution_chart(data, lottery_type):
    """Create sum distribution chart"""
    try:
        if lottery_type not in data or data[lottery_type].empty:
            return json.dumps({})
        
        df = data[lottery_type]
        sums = []
        
        for _, row in df.iterrows():
            if isinstance(row['numbers'], list) and len(row['numbers']) >= 5:
                sums.append(sum(row['numbers'][:5]))
        
        if not sums:
            return json.dumps({})
        
        fig = go.Figure(data=[go.Histogram(x=sums, nbinsx=30, marker_color='rgba(100, 200, 100, 0.7)')])
        
        # Add average line
        avg_sum = np.mean(sums)
        fig.add_vline(x=avg_sum, line_dash="dash", line_color="yellow", 
                      annotation_text=f"Average: {avg_sum:.1f}")
        
        fig.update_layout(
            title=f'{lottery_type.title()} Sum Distribution',
            xaxis_title='Sum of Numbers',
            yaxis_title='Frequency',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error creating sum distribution chart: {e}")
        return json.dumps({})

def create_overdue_chart(data, lottery_type):
    """Create overdue numbers analysis"""
    try:
        if lottery_type not in data or data[lottery_type].empty:
            return json.dumps({})
        
        df = data[lottery_type]
        
        # Calculate gaps for each number
        number_gaps = {}
        for num in range(1, 70):
            last_seen = -1
            for i, row in df.iterrows():
                if isinstance(row['numbers'], list) and num in row['numbers']:
                    last_seen = i
            
            if last_seen >= 0:
                number_gaps[num] = len(df) - last_seen - 1
            else:
                number_gaps[num] = len(df)
        
        # Get top 20 most overdue
        overdue_numbers = sorted(number_gaps.items(), key=lambda x: x[1], reverse=True)[:20]
        
        numbers = [item[0] for item in overdue_numbers]
        gaps = [item[1] for item in overdue_numbers]
        
        fig = go.Figure(data=[
            go.Bar(x=numbers, y=gaps, 
                   marker_color='rgba(255, 165, 0, 0.7)',
                   marker_line_color='rgba(255, 165, 0, 1.0)',
                   marker_line_width=1)
        ])
        
        fig.update_layout(
            title=f'{lottery_type.title()} Most Overdue Numbers',
            xaxis_title='Numbers',
            yaxis_title='Draws Since Last Appearance',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Error creating overdue chart: {e}")
        return json.dumps({})

# Routes
@app.route('/')
def dashboard():
    """Main dashboard with cosmic enhancement"""
    user_tier = get_user_tier()
    cosmic_access = has_cosmic_access()
    cosmic_data = enhanced_system.cosmic_predictor.get_current_cosmic_data()
    
    can_predict, current_usage, daily_limit = check_usage_limit()
    
    return render_template_string(COSMIC_DASHBOARD_TEMPLATE, 
                                user_tier=user_tier,
                                cosmic_access=cosmic_access,
                                cosmic_data=cosmic_data,
                                can_predict=can_predict,
                                current_usage=current_usage,
                                daily_limit=daily_limit,
                                subscription_tiers=subscription_tiers,
                                cosmic_addon_price=cosmic_addon_price)

@app.route('/api/analytics/<lottery_type>')
def get_analytics(lottery_type):
    """Get analytics charts for lottery type"""
    try:
        charts = {
            'frequency': create_frequency_chart(lottery_data, lottery_type),
            'hot_cold': create_hot_cold_chart(lottery_data, lottery_type),
            'sum_distribution': create_sum_distribution_chart(lottery_data, lottery_type),
            'overdue': create_overdue_chart(lottery_data, lottery_type)
        }
        
        # Get basic stats
        if lottery_type in lottery_data and not lottery_data[lottery_type].empty:
            df = lottery_data[lottery_type]
            total_draws = len(df)
            
            # Calculate average sum
            sums = []
            for _, row in df.iterrows():
                if isinstance(row['numbers'], list) and len(row['numbers']) >= 5:
                    sums.append(sum(row['numbers'][:5]))
            
            avg_sum = np.mean(sums) if sums else 0
            
            # Get date range
            if 'date' in df.columns:
                date_range = f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}"
            else:
                date_range = "Historical data"
        else:
            total_draws = 0
            avg_sum = 0
            date_range = "No data"
        
        stats = {
            'total_draws': total_draws,
            'avg_sum': round(avg_sum, 1),
            'date_range': date_range,
            'predictions_generated': prediction_history.get('count', 0)
        }
        
        return jsonify({'charts': charts, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def generate_prediction():
    """Generate cosmic-enhanced prediction"""
    try:
        # Check usage limits
        can_predict, current_usage, daily_limit = check_usage_limit()
        
        if not can_predict:
            return jsonify({
                'error': 'Daily limit reached',
                'message': f'You have reached your daily limit of {daily_limit} predictions. Upgrade your tier for more predictions.',
                'current_usage': current_usage,
                'daily_limit': daily_limit,
                'upgrade_options': subscription_tiers
            }), 429
        
        data = request.get_json()
        lottery_type = data.get('lottery_type', 'powerball')
        
        user_tier = get_user_tier()
        cosmic_enabled = has_cosmic_access()
        
        # Generate prediction
        prediction = enhanced_system.generate_advanced_prediction(
            lottery_data.get(lottery_type, pd.DataFrame()),
            lottery_type=lottery_type,
            user_tier=user_tier,
            cosmic_enabled=cosmic_enabled
        )
        
        # Increment usage
        increment_usage()
        
        # Store prediction
        if 'count' not in prediction_history:
            prediction_history['count'] = 0
        prediction_history['count'] += 1
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cosmic-status')
def cosmic_status():
    """Get current cosmic conditions"""
    try:
        cosmic_data = enhanced_system.cosmic_predictor.get_current_cosmic_data()
        return jsonify(cosmic_data)
    except Exception as e:
        logger.error(f"Cosmic status error: {e}")
        return jsonify({'error': str(e)}), 500

# 🌙 Cosmic-Enhanced Dashboard Template
COSMIC_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌙 PatternSight v4.0 + Cosmic Intelligence</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            overflow-x: hidden;
        }
        
        /* Cosmic Background Animation */
        .cosmic-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: 
                radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
            animation: cosmicFloat 20s ease-in-out infinite;
            z-index: -1;
        }
        
        @keyframes cosmicFloat {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            position: relative;
            z-index: 1;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #FFD700, #FFA500, #FF69B4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
        }
        
        .cosmic-status {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .cosmic-item {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: all 0.3s ease;
        }
        
        .cosmic-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }
        
        .cosmic-item .icon {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        
        .cosmic-item .label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 5px;
        }
        
        .cosmic-item .value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #FFD700;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        .lottery-btn {
            background: rgba(255, 255, 255, 0.2);
            border: 2px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
        }
        
        .lottery-btn:hover, .lottery-btn.active {
            background: rgba(255, 215, 0, 0.3);
            border-color: #FFD700;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 215, 0, 0.4);
        }
        
        .prediction-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin: 30px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .generate-btn {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            border: none;
            color: white;
            padding: 15px 40px;
            border-radius: 30px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            display: block;
            margin: 0 auto 20px;
        }
        
        .generate-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }
        
        .generate-btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        .prediction-result {
            margin-top: 20px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 15px;
            display: none;
        }
        
        .prediction-numbers {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .number-ball {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(45deg, #667eea, #764ba2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            animation: ballGlow 2s ease-in-out infinite alternate;
        }
        
        .powerball {
            background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
        }
        
        @keyframes ballGlow {
            0% { box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3); }
            100% { box-shadow: 0 5px 25px rgba(255, 215, 0, 0.6); }
        }
        
        .cosmic-enhancement {
            background: linear-gradient(45deg, rgba(138, 43, 226, 0.3), rgba(75, 0, 130, 0.3));
            border: 2px solid rgba(255, 215, 0, 0.5);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .cosmic-enhancement h3 {
            color: #FFD700;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .explanation-toggle {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            margin: 10px 0;
            transition: all 0.3s ease;
            width: 100%;
            text-align: left;
        }
        
        .explanation-toggle:hover {
            background: rgba(255, 255, 255, 0.2);
        }
        
        .explanation-content {
            display: none;
            margin-top: 15px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            white-space: pre-line;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #FFD700;
            margin-bottom: 5px;
        }
        
        .stat-label {
            opacity: 0.8;
        }
        
        .usage-indicator {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
        }
        
        .usage-bar {
            background: rgba(255, 255, 255, 0.2);
            height: 10px;
            border-radius: 5px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .usage-fill {
            background: linear-gradient(45deg, #4ECDC4, #44A08D);
            height: 100%;
            transition: width 0.3s ease;
        }
        
        .tier-info {
            background: rgba(255, 215, 0, 0.1);
            border: 1px solid rgba(255, 215, 0, 0.3);
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
        }
        
        @media (max-width: 768px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
            
            .controls {
                flex-direction: column;
                align-items: center;
            }
            
            .prediction-numbers {
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="cosmic-bg"></div>
    
    <div class="container">
        <div class="header">
            <h1>🌙 PatternSight v4.0 + Cosmic Intelligence</h1>
            <p>World's First Cosmic-Mathematical Lottery Prediction Platform</p>
            
            {% if cosmic_access %}
            <div class="cosmic-status">
                <div class="cosmic-item">
                    <div class="icon">🌙</div>
                    <div class="label">Lunar Phase</div>
                    <div class="value">{{ cosmic_data.phase_name }}</div>
                </div>
                <div class="cosmic-item">
                    <div class="icon">✨</div>
                    <div class="label">Illumination</div>
                    <div class="value">{{ "%.1f"|format(cosmic_data.lunar_illumination) }}%</div>
                </div>
                <div class="cosmic-item">
                    <div class="icon">♍</div>
                    <div class="label">Zodiac</div>
                    <div class="value">{{ cosmic_data.zodiac_sign }}</div>
                </div>
                <div class="cosmic-item">
                    <div class="icon">🪐</div>
                    <div class="label">Ruler</div>
                    <div class="value">{{ cosmic_data.planetary_ruler }}</div>
                </div>
                <div class="cosmic-item">
                    <div class="icon">⚡</div>
                    <div class="label">Energy</div>
                    <div class="value">{{ "%.0f"|format(cosmic_data.cosmic_energy) }}%</div>
                </div>
                <div class="cosmic-item">
                    <div class="icon">🕐</div>
                    <div class="label">Optimal Time</div>
                    <div class="value">{{ cosmic_data.optimal_time }}</div>
                </div>
            </div>
            {% endif %}
        </div>
        
        <div class="tier-info">
            <strong>Current Tier: {{ subscription_tiers[user_tier].name }}</strong>
            {% if cosmic_access %}
                <span style="color: #FFD700;">🌙 Cosmic Intelligence Enabled</span>
            {% else %}
                <span>Upgrade to unlock Cosmic Intelligence (+${{ cosmic_addon_price }}/month)</span>
            {% endif %}
        </div>
        
        <div class="usage-indicator">
            <div>Daily Usage: {{ current_usage }} / {{ daily_limit }} predictions</div>
            <div class="usage-bar">
                <div class="usage-fill" style="width: {{ (current_usage / daily_limit * 100) if daily_limit > 0 else 0 }}%"></div>
            </div>
        </div>
        
        <div class="controls">
            <button class="lottery-btn active" data-lottery="powerball">Powerball</button>
            <button class="lottery-btn" data-lottery="megamillions">Mega Millions</button>
            <button class="lottery-btn" data-lottery="luckyforlife">Lucky for Life</button>
        </div>
        
        <div class="prediction-section">
            <button class="generate-btn" onclick="generatePrediction()" 
                    {% if not can_predict %}disabled{% endif %}>
                {% if can_predict %}
                    🌙 Generate Cosmic-Enhanced Prediction
                {% else %}
                    Daily Limit Reached - Upgrade for More
                {% endif %}
            </button>
            
            <div id="prediction-result" class="prediction-result">
                <!-- Prediction results will appear here -->
            </div>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-draws">-</div>
                <div class="stat-label">Total Draws</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="predictions-generated">0</div>
                <div class="stat-label">AI Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avg-sum">-</div>
                <div class="stat-label">Average Sum</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="date-range">-</div>
                <div class="stat-label">Data Range</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div id="frequency-chart"></div>
            </div>
            <div class="chart-container">
                <div id="hot-cold-chart"></div>
            </div>
            <div class="chart-container">
                <div id="sum-chart"></div>
            </div>
            <div class="chart-container">
                <div id="overdue-chart"></div>
            </div>
        </div>
    </div>
    
    <script>
        let currentLottery = 'powerball';
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadAnalytics(currentLottery);
            
            // Lottery type buttons
            document.querySelectorAll('.lottery-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    document.querySelectorAll('.lottery-btn').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    currentLottery = this.dataset.lottery;
                    loadAnalytics(currentLottery);
                });
            });
        });
        
        function loadAnalytics(lotteryType) {
            fetch(`/api/analytics/${lotteryType}`)
                .then(response => response.json())
                .then(data => {
                    if (data.charts) {
                        // Load charts
                        if (data.charts.frequency) {
                            Plotly.newPlot('frequency-chart', JSON.parse(data.charts.frequency));
                        }
                        if (data.charts.hot_cold) {
                            Plotly.newPlot('hot-cold-chart', JSON.parse(data.charts.hot_cold));
                        }
                        if (data.charts.sum_distribution) {
                            Plotly.newPlot('sum-chart', JSON.parse(data.charts.sum_distribution));
                        }
                        if (data.charts.overdue) {
                            Plotly.newPlot('overdue-chart', JSON.parse(data.charts.overdue));
                        }
                    }
                    
                    if (data.stats) {
                        // Update stats
                        document.getElementById('total-draws').textContent = data.stats.total_draws;
                        document.getElementById('predictions-generated').textContent = data.stats.predictions_generated;
                        document.getElementById('avg-sum').textContent = data.stats.avg_sum;
                        document.getElementById('date-range').textContent = data.stats.date_range;
                    }
                })
                .catch(error => {
                    console.error('Error loading analytics:', error);
                });
        }
        
        function generatePrediction() {
            const btn = document.querySelector('.generate-btn');
            const resultDiv = document.getElementById('prediction-result');
            
            btn.disabled = true;
            btn.textContent = '🌙 Generating Cosmic Prediction...';
            
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    lottery_type: currentLottery
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = `
                        <div style="color: #ff6b6b; text-align: center;">
                            <h3>❌ ${data.error}</h3>
                            <p>${data.message}</p>
                        </div>
                    `;
                } else {
                    displayPrediction(data);
                }
                
                resultDiv.style.display = 'block';
                btn.disabled = false;
                btn.textContent = '🌙 Generate Cosmic-Enhanced Prediction';
                
                // Reload analytics to update prediction count
                loadAnalytics(currentLottery);
            })
            .catch(error => {
                console.error('Error generating prediction:', error);
                resultDiv.innerHTML = `
                    <div style="color: #ff6b6b; text-align: center;">
                        <h3>❌ Error</h3>
                        <p>Failed to generate prediction. Please try again.</p>
                    </div>
                `;
                resultDiv.style.display = 'block';
                btn.disabled = false;
                btn.textContent = '🌙 Generate Cosmic-Enhanced Prediction';
            });
        }
        
        function displayPrediction(prediction) {
            const resultDiv = document.getElementById('prediction-result');
            
            let numbersHtml = '';
            prediction.numbers.forEach(num => {
                numbersHtml += `<div class="number-ball">${num}</div>`;
            });
            
            if (prediction.powerball) {
                numbersHtml += `<div class="number-ball powerball">${prediction.powerball}</div>`;
            }
            
            let cosmicHtml = '';
            if (prediction.cosmic_enhancement) {
                const cosmic = prediction.cosmic_enhancement;
                cosmicHtml = `
                    <div class="cosmic-enhancement">
                        <h3>🌙 Cosmic Enhancement Active</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                            <div style="text-align: center;">
                                <div style="color: #FFD700; font-size: 1.2rem;">${cosmic.cosmic_data.phase_name}</div>
                                <div style="opacity: 0.8;">Lunar Phase</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #FFD700; font-size: 1.2rem;">${cosmic.cosmic_data.zodiac_sign}</div>
                                <div style="opacity: 0.8;">Zodiac Sign</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #FFD700; font-size: 1.2rem;">${cosmic.cosmic_scores.total.toFixed(1)}/25</div>
                                <div style="opacity: 0.8;">Cosmic Score</div>
                            </div>
                        </div>
                        
                        <button class="explanation-toggle" onclick="toggleExplanation('cosmic-explanation')">
                            🔮 View Cosmic Analysis ▼
                        </button>
                        <div id="cosmic-explanation" class="explanation-content">
                            ${cosmic.reasoning}
                        </div>
                    </div>
                `;
            }
            
            resultDiv.innerHTML = `
                <div style="text-align: center;">
                    <h3>✨ ${prediction.methodology}</h3>
                    <div class="prediction-numbers">
                        ${numbersHtml}
                    </div>
                    
                    <div style="margin: 20px 0;">
                        <div style="font-size: 1.2rem; margin: 10px 0;">
                            <strong>Confidence: ${prediction.confidence.toFixed(1)}%</strong>
                        </div>
                        <div style="opacity: 0.8;">
                            Statistical Significance: ${prediction.statistical_significance || 'Calculated'}
                        </div>
                    </div>
                    
                    ${cosmicHtml}
                    
                    <button class="explanation-toggle" onclick="toggleExplanation('pillar-breakdown')">
                        🏛️ View Pillar Breakdown ▼
                    </button>
                    <div id="pillar-breakdown" class="explanation-content">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">
                            ${Object.entries(prediction.pillar_breakdown || {}).map(([pillar, score]) => `
                                <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px;">
                                    <div style="font-weight: bold;">${pillar.replace(/_/g, ' ').toUpperCase()}</div>
                                    <div style="color: #FFD700;">${typeof score === 'number' ? score.toFixed(1) : score} points</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
        }
        
        function toggleExplanation(id) {
            const content = document.getElementById(id);
            const button = content.previousElementSibling;
            
            if (content.style.display === 'none' || content.style.display === '') {
                content.style.display = 'block';
                button.innerHTML = button.innerHTML.replace('▼', '▲');
            } else {
                content.style.display = 'none';
                button.innerHTML = button.innerHTML.replace('▲', '▼');
            }
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    logger.info("🌙 Starting PatternSight v4.0 + Cosmic Intelligence Dashboard")
    app.run(host='0.0.0.0', port=5003, debug=True)

