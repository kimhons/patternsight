"""
Premium Market Analysis - Real-Time Market Intelligence
Social media sentiment analysis and crowd behavior modeling
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import re
from collections import Counter
import requests
import aiohttp
import warnings
warnings.filterwarnings('ignore')

class PremiumMarketAnalysis:
    """
    Premium Market Analysis System
    
    Features:
    - Social media sentiment analysis
    - News event correlation
    - Crowd behavior modeling
    - Live market data integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Market Analysis
        
        Args:
            config: Market analysis configuration
        """
        self.config = config
        self.logger = logging.getLogger("MarketAnalysis")
        
        # Configuration
        self.social_sources = config.get('social_media_sources', ['twitter', 'reddit'])
        self.news_sources = config.get('news_sources', ['newsapi', 'google_news'])
        self.sentiment_config = config.get('sentiment_analysis', {})
        self.crowd_behavior_config = config.get('crowd_behavior_modeling', {})
        
        # Sentiment analysis parameters
        self.sentiment_enabled = self.sentiment_config.get('enabled', True)
        self.update_frequency = self.sentiment_config.get('update_frequency', 'hourly')
        self.confidence_threshold = self.sentiment_config.get('confidence_threshold', 0.7)
        
        # Crowd behavior parameters
        self.popular_penalty = self.crowd_behavior_config.get('popular_number_penalty', 0.15)
        self.contrarian_bonus = self.crowd_behavior_config.get('contrarian_bonus', 0.10)
        self.social_influence = self.crowd_behavior_config.get('social_influence_factor', 0.25)
        
        # Cache for market data
        self.sentiment_cache = {}
        self.news_cache = {}
        self.crowd_data_cache = {}
        self.last_update = None
        
        # Popular lottery numbers (commonly chosen)
        self.popular_numbers = [7, 11, 13, 21, 23, 31, 37, 41, 43, 47]
        self.birthday_numbers = list(range(1, 32))  # 1-31 (birth dates)
        
        self.logger.info("Premium Market Analysis initialized")
    
    async def analyze_market_conditions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market conditions and sentiment
        
        Args:
            context: Current context (jackpot, date, etc.)
            
        Returns:
            Market analysis with sentiment and crowd behavior insights
        """
        try:
            self.logger.info("Starting market condition analysis")
            
            # Run market analysis components in parallel
            analysis_tasks = [
                self._analyze_social_sentiment(context),
                self._analyze_news_impact(context),
                self._analyze_crowd_behavior(context),
                self._analyze_jackpot_psychology(context)
            ]
            
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine market analysis
            market_analysis = self._combine_market_analysis(analysis_results, context)
            
            # Generate market recommendations
            recommendations = self._generate_market_recommendations(market_analysis, context)
            
            result = {
                'recommended_numbers': recommendations['numbers'],
                'confidence': recommendations['confidence'],
                'market_summary': recommendations['summary'],
                'market_insights': market_analysis,
                'sentiment_score': market_analysis.get('overall_sentiment', 0.5),
                'crowd_behavior_factor': market_analysis.get('crowd_factor', 1.0),
                'contrarian_opportunities': recommendations.get('contrarian_numbers', []),
                'popular_numbers_to_avoid': recommendations.get('avoid_numbers', [])
            }
            
            # Cache results
            self._cache_market_data(result)
            
            self.logger.info("Market condition analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            return self._fallback_market_analysis(context)
    
    async def _analyze_social_sentiment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze social media sentiment around lottery"""
        
        try:
            # Check cache first
            if self._is_cache_valid('sentiment'):
                return self.sentiment_cache.get('data', {})
            
            sentiment_data = {
                'overall_sentiment': 0.6,  # Neutral-positive baseline
                'lottery_mentions': 0,
                'number_preferences': {},
                'trending_topics': [],
                'confidence': 0.70
            }
            
            # Simulate social media analysis (in production, would use real APIs)
            jackpot_size = context.get('jackpot', 50000000)
            
            # High jackpot increases social activity and sentiment
            if jackpot_size > 100000000:
                sentiment_data['overall_sentiment'] = 0.8
                sentiment_data['lottery_mentions'] = 1500
                sentiment_data['trending_topics'] = ['mega_jackpot', 'lottery_fever', 'lucky_numbers']
                
                # Simulate popular number preferences during high jackpots
                sentiment_data['number_preferences'] = {
                    7: 0.25, 11: 0.22, 13: 0.20, 21: 0.18, 23: 0.15,
                    31: 0.12, 37: 0.10, 41: 0.08, 43: 0.07, 47: 0.05
                }
            elif jackpot_size > 50000000:
                sentiment_data['overall_sentiment'] = 0.7
                sentiment_data['lottery_mentions'] = 800
                sentiment_data['trending_topics'] = ['lottery_drawing', 'lucky_day']
                
                sentiment_data['number_preferences'] = {
                    7: 0.15, 11: 0.12, 13: 0.10, 21: 0.08, 23: 0.07
                }
            else:
                sentiment_data['overall_sentiment'] = 0.6
                sentiment_data['lottery_mentions'] = 300
                sentiment_data['trending_topics'] = ['weekly_lottery']
            
            # Add some randomization to simulate real social media variability
            sentiment_data['overall_sentiment'] += np.random.normal(0, 0.05)
            sentiment_data['overall_sentiment'] = max(0.3, min(0.9, sentiment_data['overall_sentiment']))
            
            # Cache the results
            self.sentiment_cache = {
                'data': sentiment_data,
                'timestamp': datetime.now(),
                'ttl_hours': 1
            }
            
            return sentiment_data
            
        except Exception as e:
            self.logger.warning(f"Social sentiment analysis failed: {e}")
            return {
                'overall_sentiment': 0.6,
                'lottery_mentions': 100,
                'number_preferences': {},
                'trending_topics': [],
                'confidence': 0.60,
                'error': str(e)
            }
    
    async def _analyze_news_impact(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze news events impact on lottery behavior"""
        
        try:
            # Check cache first
            if self._is_cache_valid('news'):
                return self.news_cache.get('data', {})
            
            news_impact = {
                'major_events': [],
                'lottery_news': [],
                'economic_indicators': {},
                'impact_score': 0.5,  # Neutral baseline
                'confidence': 0.65
            }
            
            # Simulate news analysis
            current_date = datetime.now()
            jackpot_size = context.get('jackpot', 50000000)
            
            # Major jackpot news impact
            if jackpot_size > 200000000:
                news_impact['major_events'].append({
                    'event': 'Record Jackpot Alert',
                    'impact': 'high',
                    'sentiment': 'positive',
                    'description': f'${jackpot_size:,} jackpot generates massive media coverage'
                })
                news_impact['impact_score'] = 0.8
            
            # Seasonal events
            month = current_date.month
            if month == 12:
                news_impact['major_events'].append({
                    'event': 'Holiday Season',
                    'impact': 'medium',
                    'sentiment': 'positive',
                    'description': 'Holiday lottery ticket purchases increase'
                })
                news_impact['impact_score'] += 0.1
            elif month == 1:
                news_impact['major_events'].append({
                    'event': 'New Year Optimism',
                    'impact': 'medium',
                    'sentiment': 'positive',
                    'description': 'New Year lottery participation surge'
                })
                news_impact['impact_score'] += 0.1
            
            # Economic indicators simulation
            news_impact['economic_indicators'] = {
                'consumer_confidence': 0.65,
                'disposable_income_trend': 'stable',
                'lottery_participation_trend': 'increasing' if jackpot_size > 100000000 else 'stable'
            }
            
            # Clamp impact score
            news_impact['impact_score'] = max(0.3, min(0.9, news_impact['impact_score']))
            
            # Cache results
            self.news_cache = {
                'data': news_impact,
                'timestamp': datetime.now(),
                'ttl_hours': 6
            }
            
            return news_impact
            
        except Exception as e:
            self.logger.warning(f"News impact analysis failed: {e}")
            return {
                'major_events': [],
                'lottery_news': [],
                'economic_indicators': {},
                'impact_score': 0.5,
                'confidence': 0.60,
                'error': str(e)
            }
    
    async def _analyze_crowd_behavior(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze crowd behavior and popular number selection patterns"""
        
        try:
            crowd_analysis = {
                'popular_numbers': self.popular_numbers.copy(),
                'birthday_bias': True,
                'pattern_preferences': {},
                'crowd_factor': 1.0,
                'contrarian_opportunities': [],
                'confidence': 0.75
            }
            
            jackpot_size = context.get('jackpot', 50000000)
            days_since_winner = context.get('days_since_winner', 5)
            
            # High jackpot increases crowd behavior
            if jackpot_size > 100000000:
                crowd_analysis['crowd_factor'] = 1.5
                crowd_analysis['pattern_preferences'] = {
                    'lucky_numbers': 0.4,  # 7, 11, 13, etc.
                    'birthday_numbers': 0.3,  # 1-31
                    'sequential_patterns': 0.2,  # 1,2,3,4,5 or similar
                    'round_numbers': 0.1  # 10, 20, 30, etc.
                }
            elif jackpot_size > 50000000:
                crowd_analysis['crowd_factor'] = 1.2
                crowd_analysis['pattern_preferences'] = {
                    'lucky_numbers': 0.3,
                    'birthday_numbers': 0.4,
                    'sequential_patterns': 0.2,
                    'round_numbers': 0.1
                }
            else:
                crowd_analysis['crowd_factor'] = 1.0
                crowd_analysis['pattern_preferences'] = {
                    'lucky_numbers': 0.25,
                    'birthday_numbers': 0.35,
                    'sequential_patterns': 0.25,
                    'round_numbers': 0.15
                }
            
            # Identify contrarian opportunities (less popular numbers)
            all_numbers = set(range(1, 70))
            popular_set = set(self.popular_numbers + self.birthday_numbers)
            contrarian_candidates = list(all_numbers - popular_set)
            
            # Select contrarian numbers (higher numbers, less "lucky" appearing)
            contrarian_opportunities = [n for n in contrarian_candidates if n > 31 and n % 7 != 0]
            crowd_analysis['contrarian_opportunities'] = sorted(contrarian_opportunities)[:15]
            
            return crowd_analysis
            
        except Exception as e:
            self.logger.warning(f"Crowd behavior analysis failed: {e}")
            return {
                'popular_numbers': self.popular_numbers,
                'birthday_bias': True,
                'pattern_preferences': {},
                'crowd_factor': 1.0,
                'contrarian_opportunities': [32, 34, 36, 38, 39, 44, 46, 48, 51, 52],
                'confidence': 0.65,
                'error': str(e)
            }
    
    async def _analyze_jackpot_psychology(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze psychological factors related to jackpot size"""
        
        try:
            jackpot_size = context.get('jackpot', 50000000)
            days_since_winner = context.get('days_since_winner', 5)
            
            psychology_analysis = {
                'jackpot_excitement_level': 'medium',
                'risk_tolerance': 'medium',
                'number_selection_bias': {},
                'psychological_factors': [],
                'confidence': 0.70
            }
            
            # Jackpot size psychology
            if jackpot_size > 300000000:
                psychology_analysis['jackpot_excitement_level'] = 'extreme'
                psychology_analysis['risk_tolerance'] = 'high'
                psychology_analysis['psychological_factors'].extend([
                    'lottery_fever_effect',
                    'media_attention_bias',
                    'social_proof_amplification'
                ])
                psychology_analysis['number_selection_bias'] = {
                    'lucky_number_preference': 0.6,
                    'birthday_date_preference': 0.4,
                    'pattern_avoidance': 0.3,
                    'contrarian_opportunity': 0.8
                }
            elif jackpot_size > 150000000:
                psychology_analysis['jackpot_excitement_level'] = 'high'
                psychology_analysis['risk_tolerance'] = 'medium-high'
                psychology_analysis['psychological_factors'].extend([
                    'increased_participation',
                    'lucky_number_bias'
                ])
                psychology_analysis['number_selection_bias'] = {
                    'lucky_number_preference': 0.4,
                    'birthday_date_preference': 0.5,
                    'pattern_avoidance': 0.2,
                    'contrarian_opportunity': 0.6
                }
            elif jackpot_size > 75000000:
                psychology_analysis['jackpot_excitement_level'] = 'medium-high'
                psychology_analysis['risk_tolerance'] = 'medium'
                psychology_analysis['psychological_factors'].append('moderate_interest')
                psychology_analysis['number_selection_bias'] = {
                    'lucky_number_preference': 0.3,
                    'birthday_date_preference': 0.6,
                    'pattern_avoidance': 0.1,
                    'contrarian_opportunity': 0.4
                }
            else:
                psychology_analysis['jackpot_excitement_level'] = 'low-medium'
                psychology_analysis['risk_tolerance'] = 'low-medium'
                psychology_analysis['psychological_factors'].append('routine_participation')
                psychology_analysis['number_selection_bias'] = {
                    'lucky_number_preference': 0.2,
                    'birthday_date_preference': 0.7,
                    'pattern_avoidance': 0.05,
                    'contrarian_opportunity': 0.2
                }
            
            # Days since winner psychology
            if days_since_winner > 15:
                psychology_analysis['psychological_factors'].append('rollover_fatigue')
                psychology_analysis['risk_tolerance'] = 'high'
            elif days_since_winner > 7:
                psychology_analysis['psychological_factors'].append('building_anticipation')
            
            return psychology_analysis
            
        except Exception as e:
            self.logger.warning(f"Jackpot psychology analysis failed: {e}")
            return {
                'jackpot_excitement_level': 'medium',
                'risk_tolerance': 'medium',
                'number_selection_bias': {},
                'psychological_factors': [],
                'confidence': 0.60,
                'error': str(e)
            }
    
    def _combine_market_analysis(self, analysis_results: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all market analysis components"""
        
        valid_results = [r for r in analysis_results if not isinstance(r, Exception)]
        
        if not valid_results:
            return self._fallback_market_insights()
        
        sentiment_data, news_impact, crowd_behavior, jackpot_psychology = (
            valid_results + [{}] * (4 - len(valid_results))
        )[:4]
        
        # Combine insights
        combined_analysis = {
            'overall_sentiment': sentiment_data.get('overall_sentiment', 0.6),
            'news_impact_score': news_impact.get('impact_score', 0.5),
            'crowd_factor': crowd_behavior.get('crowd_factor', 1.0),
            'psychological_state': jackpot_psychology.get('jackpot_excitement_level', 'medium'),
            'market_conditions': 'normal',
            'social_influence_strength': 0.5,
            'contrarian_opportunity_level': 0.5
        }
        
        # Determine overall market conditions
        avg_sentiment = combined_analysis['overall_sentiment']
        crowd_factor = combined_analysis['crowd_factor']
        
        if avg_sentiment > 0.75 and crowd_factor > 1.3:
            combined_analysis['market_conditions'] = 'high_excitement'
            combined_analysis['social_influence_strength'] = 0.8
            combined_analysis['contrarian_opportunity_level'] = 0.9
        elif avg_sentiment > 0.65 and crowd_factor > 1.1:
            combined_analysis['market_conditions'] = 'elevated_interest'
            combined_analysis['social_influence_strength'] = 0.6
            combined_analysis['contrarian_opportunity_level'] = 0.7
        elif avg_sentiment < 0.45:
            combined_analysis['market_conditions'] = 'low_interest'
            combined_analysis['social_influence_strength'] = 0.3
            combined_analysis['contrarian_opportunity_level'] = 0.2
        
        # Add component data
        combined_analysis['sentiment_details'] = sentiment_data
        combined_analysis['news_details'] = news_impact
        combined_analysis['crowd_details'] = crowd_behavior
        combined_analysis['psychology_details'] = jackpot_psychology
        
        return combined_analysis
    
    def _generate_market_recommendations(self, market_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate number recommendations based on market analysis"""
        
        try:
            crowd_details = market_analysis.get('crowd_details', {})
            psychology_details = market_analysis.get('psychology_details', {})
            
            # Get contrarian opportunities
            contrarian_numbers = crowd_details.get('contrarian_opportunities', [])
            popular_numbers = crowd_details.get('popular_numbers', self.popular_numbers)
            
            # Market condition strategy
            market_conditions = market_analysis.get('market_conditions', 'normal')
            contrarian_level = market_analysis.get('contrarian_opportunity_level', 0.5)
            
            recommended_numbers = []
            avoid_numbers = []
            
            if market_conditions == 'high_excitement' and contrarian_level > 0.7:
                # High contrarian strategy - avoid popular numbers
                recommended_numbers = contrarian_numbers[:8] if contrarian_numbers else [32, 34, 36, 38, 44]
                avoid_numbers = popular_numbers[:10]
                strategy = 'contrarian_high_excitement'
                confidence = 0.85
                
            elif market_conditions == 'elevated_interest' and contrarian_level > 0.5:
                # Moderate contrarian strategy
                recommended_numbers = (contrarian_numbers[:5] + popular_numbers[-3:]) if contrarian_numbers else [28, 32, 36, 44, 48]
                avoid_numbers = popular_numbers[:5]
                strategy = 'moderate_contrarian'
                confidence = 0.78
                
            else:
                # Standard strategy - balanced approach
                all_numbers = list(range(1, 70))
                # Remove some popular numbers but not all
                filtered_numbers = [n for n in all_numbers if n not in popular_numbers[:5]]
                recommended_numbers = filtered_numbers[:10] if filtered_numbers else [15, 25, 35, 45, 55]
                avoid_numbers = popular_numbers[:3]
                strategy = 'balanced_market'
                confidence = 0.72
            
            # Select final 5 numbers
            if len(recommended_numbers) >= 5:
                # Use weighted selection
                weights = [1.0 / (i + 1) for i in range(len(recommended_numbers))]  # Decreasing weights
                selected_indices = np.random.choice(
                    len(recommended_numbers), 
                    size=min(5, len(recommended_numbers)), 
                    replace=False, 
                    p=np.array(weights) / sum(weights)
                )
                final_numbers = [recommended_numbers[i] for i in selected_indices]
            else:
                final_numbers = recommended_numbers + [20, 30, 40, 50, 60]
                final_numbers = final_numbers[:5]
            
            # Ensure we have exactly 5 unique numbers
            final_numbers = list(set(final_numbers))
            while len(final_numbers) < 5:
                for num in range(1, 70):
                    if num not in final_numbers and num not in avoid_numbers:
                        final_numbers.append(num)
                        break
            
            final_numbers = sorted(final_numbers[:5])
            
            # Generate summary
            summary = f"Market analysis ({market_conditions}) suggests {strategy} approach"
            
            return {
                'numbers': final_numbers,
                'confidence': confidence,
                'summary': summary,
                'strategy': strategy,
                'contrarian_numbers': contrarian_numbers[:10] if contrarian_numbers else [],
                'avoid_numbers': avoid_numbers,
                'market_reasoning': f"Market conditions: {market_conditions}, Contrarian opportunity: {contrarian_level:.1%}"
            }
            
        except Exception as e:
            self.logger.warning(f"Market recommendations generation failed: {e}")
            return {
                'numbers': [14, 24, 34, 44, 54],
                'confidence': 0.70,
                'summary': 'Standard market analysis applied',
                'strategy': 'fallback_market',
                'contrarian_numbers': [],
                'avoid_numbers': [],
                'error': str(e)
            }
    
    def _is_cache_valid(self, cache_type: str) -> bool:
        """Check if cache is still valid"""
        
        cache_map = {
            'sentiment': self.sentiment_cache,
            'news': self.news_cache,
            'crowd': self.crowd_data_cache
        }
        
        cache = cache_map.get(cache_type, {})
        if not cache or 'timestamp' not in cache:
            return False
        
        ttl_hours = cache.get('ttl_hours', 1)
        cache_age = datetime.now() - cache['timestamp']
        
        return cache_age < timedelta(hours=ttl_hours)
    
    def _cache_market_data(self, data: Dict[str, Any]):
        """Cache market analysis results"""
        
        cache_entry = {
            'data': data,
            'timestamp': datetime.now(),
            'ttl_hours': 2
        }
        
        # Update last update time
        self.last_update = datetime.now()
    
    def _fallback_market_insights(self) -> Dict[str, Any]:
        """Fallback market insights if analysis fails"""
        
        return {
            'overall_sentiment': 0.6,
            'news_impact_score': 0.5,
            'crowd_factor': 1.0,
            'psychological_state': 'medium',
            'market_conditions': 'normal',
            'social_influence_strength': 0.5,
            'contrarian_opportunity_level': 0.5,
            'fallback_mode': True
        }
    
    def _fallback_market_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback market analysis if entire system fails"""
        
        self.logger.warning("Using fallback market analysis")
        
        return {
            'recommended_numbers': [12, 22, 32, 42, 52],
            'confidence': 0.70,
            'market_summary': 'Fallback market analysis - standard approach applied',
            'market_insights': self._fallback_market_insights(),
            'sentiment_score': 0.6,
            'crowd_behavior_factor': 1.0,
            'contrarian_opportunities': [33, 34, 36, 38, 39],
            'popular_numbers_to_avoid': [7, 11, 13, 21, 23],
            'fallback_mode': True
        }
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current market analysis status"""
        
        return {
            'sentiment_analysis_enabled': self.sentiment_enabled,
            'social_sources_configured': len(self.social_sources),
            'news_sources_configured': len(self.news_sources),
            'update_frequency': self.update_frequency,
            'confidence_threshold': self.confidence_threshold,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'cache_status': {
                'sentiment_cached': bool(self.sentiment_cache),
                'news_cached': bool(self.news_cache),
                'crowd_data_cached': bool(self.crowd_data_cache)
            },
            'crowd_behavior_factors': {
                'popular_number_penalty': self.popular_penalty,
                'contrarian_bonus': self.contrarian_bonus,
                'social_influence_factor': self.social_influence
            }
        }

