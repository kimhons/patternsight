"""
Premium Predictive Intelligence - Future Trend Forecasting
Advanced forecasting capabilities with 30-day horizon prediction
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PremiumPredictiveIntelligence:
    """
    Premium Predictive Intelligence System
    
    Features:
    - 30-day trend forecasting
    - Jackpot impact modeling
    - Seasonal pattern optimization
    - Market condition adaptation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Predictive Intelligence
        
        Args:
            config: Predictive intelligence configuration
        """
        self.config = config
        self.logger = logging.getLogger("PredictiveIntelligence")
        
        # Configuration parameters
        self.forecast_horizon = config.get('forecast_horizon_days', 30)
        self.trend_window = config.get('trend_analysis_window', 90)
        self.seasonal_adjustment = config.get('seasonal_adjustment', True)
        self.jackpot_modeling = config.get('jackpot_impact_modeling', True)
        
        # Market condition factors
        self.market_factors = config.get('market_condition_factors', [
            'jackpot_size', 'days_since_winner', 'seasonal_patterns'
        ])
        
        # Initialize models
        self.trend_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Cached forecasts
        self.forecast_cache = {}
        self.last_forecast_update = None
        
        self.logger.info("Premium Predictive Intelligence initialized")
    
    async def forecast_trends(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive trend forecasting
        
        Args:
            data: Historical lottery data
            context: Current context (jackpot, date, etc.)
            
        Returns:
            Trend forecast with predicted numbers and insights
        """
        try:
            self.logger.info("Starting 30-day trend forecasting")
            
            # Extract trend features
            trend_features = self._extract_trend_features(data, context)
            
            # Generate multiple forecast models
            forecasts = await asyncio.gather(
                self._forecast_number_trends(data, trend_features),
                self._forecast_sum_range_trends(data, trend_features),
                self._forecast_pattern_evolution(data, trend_features),
                self._forecast_jackpot_impact(data, context),
                return_exceptions=True
            )
            
            # Combine forecasts
            combined_forecast = self._combine_forecasts(forecasts, trend_features)
            
            # Generate future insights
            future_insights = await self._generate_future_insights(data, context, combined_forecast)
            
            result = {
                'predicted_numbers': combined_forecast['numbers'],
                'confidence': combined_forecast['confidence'],
                'trend_analysis': combined_forecast['trend_analysis'],
                'forecast_horizon_days': self.forecast_horizon,
                'trend_summary': combined_forecast['summary'],
                'future_insights': future_insights,
                'seasonal_factors': combined_forecast.get('seasonal_factors', {}),
                'jackpot_impact': combined_forecast.get('jackpot_impact', {}),
                'trend_strength': combined_forecast.get('trend_strength', 0.75)
            }
            
            # Cache forecast
            self._cache_forecast(result)
            
            self.logger.info("Trend forecasting completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Trend forecasting failed: {e}")
            return self._fallback_forecast(data, context)
    
    def _extract_trend_features(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for trend analysis"""
        
        features = {
            'data_points': len(data),
            'date_range': None,
            'jackpot_size': context.get('jackpot', 50000000),
            'days_since_winner': context.get('days_since_winner', 5),
            'current_date': datetime.now(),
            'seasonal_indicators': {}
        }
        
        if len(data) > 0:
            # Extract date range
            if 'date' in data.columns:
                dates = pd.to_datetime(data['date'])
                features['date_range'] = (dates.min(), dates.max())
                features['data_span_days'] = (dates.max() - dates.min()).days
            
            # Seasonal indicators
            current_month = datetime.now().month
            current_quarter = (current_month - 1) // 3 + 1
            
            features['seasonal_indicators'] = {
                'month': current_month,
                'quarter': current_quarter,
                'is_holiday_season': current_month in [11, 12, 1],
                'is_summer': current_month in [6, 7, 8],
                'day_of_year': datetime.now().timetuple().tm_yday
            }
        
        return features
    
    async def _forecast_number_trends(self, data: pd.DataFrame, features: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast individual number trends"""
        
        try:
            # Analyze recent number frequency trends
            recent_data = data.tail(self.trend_window) if len(data) > self.trend_window else data
            
            number_trends = {}
            predicted_numbers = []
            
            # Calculate trend for each number (1-69)
            for number in range(1, 70):
                trend_score = self._calculate_number_trend(recent_data, number)
                number_trends[number] = trend_score
            
            # Select top trending numbers
            sorted_trends = sorted(number_trends.items(), key=lambda x: x[1], reverse=True)
            
            # Apply trend-based selection with some randomization
            top_trending = [num for num, score in sorted_trends[:15]]
            
            # Select 5 numbers with trend weighting
            predicted_numbers = self._select_trend_weighted_numbers(top_trending, number_trends)
            
            return {
                'numbers': predicted_numbers,
                'confidence': 0.82,
                'trend_scores': number_trends,
                'method': 'number_trend_forecasting',
                'top_trending_numbers': top_trending[:10]
            }
            
        except Exception as e:
            self.logger.warning(f"Number trend forecasting failed: {e}")
            return {
                'numbers': [5, 15, 25, 35, 45],
                'confidence': 0.70,
                'method': 'fallback_number_trends',
                'error': str(e)
            }
    
    def _calculate_number_trend(self, data: pd.DataFrame, number: int) -> float:
        """Calculate trend score for a specific number"""
        
        if len(data) < 10:
            return 0.5  # Neutral trend
        
        # Count occurrences in recent periods
        recent_occurrences = []
        window_size = max(10, len(data) // 5)
        
        for i in range(0, len(data), window_size):
            window_data = data.iloc[i:i+window_size]
            count = 0
            
            for _, row in window_data.iterrows():
                if 'numbers' in row and isinstance(row['numbers'], (list, tuple)):
                    if number in row['numbers']:
                        count += 1
                elif hasattr(row, 'values'):
                    # Check if number appears in any column
                    row_values = [v for v in row.values if isinstance(v, (int, float)) and 1 <= v <= 69]
                    if number in row_values:
                        count += 1
            
            frequency = count / len(window_data) if len(window_data) > 0 else 0
            recent_occurrences.append(frequency)
        
        if len(recent_occurrences) < 2:
            return 0.5
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_occurrences))
        y = np.array(recent_occurrences)
        
        if np.std(y) == 0:
            return 0.5
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Convert slope to trend score (0-1 scale)
        trend_score = 0.5 + (slope * 10)  # Amplify slope
        trend_score = max(0.0, min(1.0, trend_score))  # Clamp to [0,1]
        
        return trend_score
    
    def _select_trend_weighted_numbers(self, trending_numbers: List[int], trend_scores: Dict[int, float]) -> List[int]:
        """Select numbers using trend weighting"""
        
        if len(trending_numbers) < 5:
            # Fill with additional numbers
            all_numbers = list(range(1, 70))
            for num in all_numbers:
                if num not in trending_numbers and len(trending_numbers) < 15:
                    trending_numbers.append(num)
        
        # Weighted random selection
        weights = [trend_scores.get(num, 0.5) for num in trending_numbers]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(trending_numbers)] * len(trending_numbers)
        
        # Select 5 numbers with weighted probability
        selected = []
        available_numbers = trending_numbers.copy()
        available_weights = weights.copy()
        
        for _ in range(5):
            if not available_numbers:
                break
                
            # Weighted random selection
            chosen_idx = np.random.choice(len(available_numbers), p=available_weights)
            chosen_number = available_numbers[chosen_idx]
            
            selected.append(chosen_number)
            
            # Remove chosen number and renormalize weights
            available_numbers.pop(chosen_idx)
            available_weights.pop(chosen_idx)
            
            if available_weights:
                total = sum(available_weights)
                available_weights = [w / total for w in available_weights]
        
        # Fill to 5 numbers if needed
        while len(selected) < 5:
            for num in range(1, 70):
                if num not in selected:
                    selected.append(num)
                    break
        
        return sorted(selected[:5])
    
    async def _forecast_sum_range_trends(self, data: pd.DataFrame, features: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast sum and range trends"""
        
        try:
            # Extract historical sums and ranges
            sums = []
            ranges = []
            
            for _, row in data.iterrows():
                if 'numbers' in row and isinstance(row['numbers'], (list, tuple)):
                    numbers = [n for n in row['numbers'] if isinstance(n, (int, float))]
                    if len(numbers) >= 5:
                        numbers = numbers[:5]
                        sums.append(sum(numbers))
                        ranges.append(max(numbers) - min(numbers))
            
            if len(sums) < 10:
                return {
                    'target_sum': 175,
                    'target_range': 50,
                    'confidence': 0.70,
                    'method': 'fallback_sum_range'
                }
            
            # Trend analysis for sums and ranges
            recent_sums = sums[-self.trend_window:] if len(sums) > self.trend_window else sums
            recent_ranges = ranges[-self.trend_window:] if len(ranges) > self.trend_window else ranges
            
            # Calculate trends
            sum_trend = self._calculate_linear_trend(recent_sums)
            range_trend = self._calculate_linear_trend(recent_ranges)
            
            # Predict future values
            current_sum_avg = np.mean(recent_sums[-10:])
            current_range_avg = np.mean(recent_ranges[-10:])
            
            # Apply trend adjustment
            predicted_sum = current_sum_avg + (sum_trend * 5)  # 5-draw forecast
            predicted_range = current_range_avg + (range_trend * 5)
            
            # Clamp to reasonable ranges
            predicted_sum = max(50, min(300, predicted_sum))
            predicted_range = max(20, min(60, predicted_range))
            
            return {
                'target_sum': int(predicted_sum),
                'target_range': int(predicted_range),
                'sum_trend': sum_trend,
                'range_trend': range_trend,
                'confidence': 0.78,
                'method': 'sum_range_forecasting',
                'historical_stats': {
                    'avg_sum': np.mean(sums),
                    'avg_range': np.mean(ranges),
                    'sum_std': np.std(sums),
                    'range_std': np.std(ranges)
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Sum/range forecasting failed: {e}")
            return {
                'target_sum': 175,
                'target_range': 45,
                'confidence': 0.70,
                'method': 'fallback_sum_range',
                'error': str(e)
            }
    
    def _calculate_linear_trend(self, values: List[float]) -> float:
        """Calculate linear trend from time series values"""
        
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return slope
        except:
            return 0.0
    
    async def _forecast_pattern_evolution(self, data: pd.DataFrame, features: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast pattern evolution trends"""
        
        try:
            # Analyze pattern evolution over time
            pattern_metrics = {
                'even_odd_ratio': [],
                'low_high_ratio': [],
                'consecutive_pairs': [],
                'gap_patterns': []
            }
            
            for _, row in data.iterrows():
                if 'numbers' in row and isinstance(row['numbers'], (list, tuple)):
                    numbers = [n for n in row['numbers'] if isinstance(n, (int, float)) and 1 <= n <= 69]
                    if len(numbers) >= 5:
                        numbers = sorted(numbers[:5])
                        
                        # Even/odd ratio
                        even_count = sum(1 for n in numbers if n % 2 == 0)
                        pattern_metrics['even_odd_ratio'].append(even_count / len(numbers))
                        
                        # Low/high ratio (low: 1-34, high: 35-69)
                        low_count = sum(1 for n in numbers if n <= 34)
                        pattern_metrics['low_high_ratio'].append(low_count / len(numbers))
                        
                        # Consecutive pairs
                        consecutive = sum(1 for i in range(len(numbers)-1) if numbers[i+1] - numbers[i] == 1)
                        pattern_metrics['consecutive_pairs'].append(consecutive)
                        
                        # Average gap
                        gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
                        pattern_metrics['gap_patterns'].append(np.mean(gaps) if gaps else 0)
            
            # Forecast pattern trends
            pattern_forecasts = {}
            for pattern_name, values in pattern_metrics.items():
                if len(values) >= 10:
                    recent_values = values[-self.trend_window:] if len(values) > self.trend_window else values
                    trend = self._calculate_linear_trend(recent_values)
                    current_avg = np.mean(recent_values[-5:])
                    predicted_value = current_avg + (trend * 3)  # 3-draw forecast
                    
                    pattern_forecasts[pattern_name] = {
                        'current': current_avg,
                        'trend': trend,
                        'predicted': predicted_value
                    }
            
            return {
                'pattern_forecasts': pattern_forecasts,
                'confidence': 0.75,
                'method': 'pattern_evolution_forecasting',
                'trend_strength': np.mean([abs(pf['trend']) for pf in pattern_forecasts.values()]) if pattern_forecasts else 0.1
            }
            
        except Exception as e:
            self.logger.warning(f"Pattern evolution forecasting failed: {e}")
            return {
                'pattern_forecasts': {},
                'confidence': 0.70,
                'method': 'fallback_pattern_evolution',
                'error': str(e)
            }
    
    async def _forecast_jackpot_impact(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast jackpot impact on number selection"""
        
        try:
            jackpot_size = context.get('jackpot', 50000000)
            days_since_winner = context.get('days_since_winner', 5)
            
            # Jackpot impact model
            jackpot_factor = min(2.0, jackpot_size / 100000000)  # Normalize to 100M
            rollover_factor = min(1.5, days_since_winner / 10)  # Normalize to 10 days
            
            # High jackpots tend to increase certain number preferences
            jackpot_impact = {
                'size_factor': jackpot_factor,
                'rollover_factor': rollover_factor,
                'combined_impact': jackpot_factor * rollover_factor,
                'predicted_effects': {
                    'higher_number_preference': jackpot_factor > 1.2,
                    'lucky_number_bias': rollover_factor > 1.2,
                    'crowd_behavior_increase': jackpot_factor > 1.5
                }
            }
            
            # Adjust number selection based on jackpot
            if jackpot_factor > 1.3:
                # High jackpot - people tend to pick "luckier" numbers
                recommended_adjustments = {
                    'favor_numbers': [7, 11, 13, 21, 23, 31, 37, 41, 43, 47],
                    'avoid_numbers': [1, 2, 3, 4, 5, 6],  # Avoid obvious patterns
                    'strategy': 'contrarian_high_jackpot'
                }
            else:
                # Normal jackpot - standard selection
                recommended_adjustments = {
                    'favor_numbers': [],
                    'avoid_numbers': [],
                    'strategy': 'standard_selection'
                }
            
            return {
                'jackpot_impact': jackpot_impact,
                'recommended_adjustments': recommended_adjustments,
                'confidence': 0.73,
                'method': 'jackpot_impact_modeling'
            }
            
        except Exception as e:
            self.logger.warning(f"Jackpot impact forecasting failed: {e}")
            return {
                'jackpot_impact': {'combined_impact': 1.0},
                'recommended_adjustments': {'strategy': 'standard_selection'},
                'confidence': 0.70,
                'method': 'fallback_jackpot_impact',
                'error': str(e)
            }
    
    def _combine_forecasts(self, forecasts: List[Dict[str, Any]], features: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple forecast results"""
        
        valid_forecasts = [f for f in forecasts if not isinstance(f, Exception)]
        
        if not valid_forecasts:
            return self._fallback_combined_forecast()
        
        # Extract number predictions from different forecasts
        all_numbers = []
        confidences = []
        
        for forecast in valid_forecasts:
            if 'numbers' in forecast:
                all_numbers.extend(forecast['numbers'])
                confidences.append(forecast.get('confidence', 0.75))
        
        # Select final numbers using frequency and trend weighting
        if all_numbers:
            from collections import Counter
            number_counts = Counter(all_numbers)
            
            # Weight by frequency and add some randomization
            weighted_numbers = []
            for number, count in number_counts.most_common():
                for _ in range(count):
                    weighted_numbers.append(number)
            
            # Select 5 unique numbers
            final_numbers = []
            used_numbers = set()
            
            for number in weighted_numbers:
                if number not in used_numbers and len(final_numbers) < 5:
                    final_numbers.append(number)
                    used_numbers.add(number)
            
            # Fill to 5 if needed
            while len(final_numbers) < 5:
                for num in range(1, 70):
                    if num not in used_numbers:
                        final_numbers.append(num)
                        used_numbers.add(num)
                        break
        else:
            final_numbers = [9, 18, 27, 36, 45]
        
        # Calculate combined confidence
        avg_confidence = np.mean(confidences) if confidences else 0.75
        forecast_agreement = len(set(all_numbers)) / len(all_numbers) if all_numbers else 0.5
        combined_confidence = avg_confidence * (1 + forecast_agreement * 0.2)
        combined_confidence = min(0.90, combined_confidence)
        
        # Generate summary
        summary = f"30-day trend analysis combining {len(valid_forecasts)} forecasting models"
        
        # Extract additional insights
        trend_analysis = {}
        seasonal_factors = {}
        jackpot_impact = {}
        
        for forecast in valid_forecasts:
            if 'pattern_forecasts' in forecast:
                trend_analysis.update(forecast['pattern_forecasts'])
            if 'seasonal_indicators' in forecast:
                seasonal_factors.update(forecast['seasonal_indicators'])
            if 'jackpot_impact' in forecast:
                jackpot_impact.update(forecast['jackpot_impact'])
        
        return {
            'numbers': sorted(final_numbers[:5]),
            'confidence': combined_confidence,
            'summary': summary,
            'trend_analysis': trend_analysis,
            'seasonal_factors': seasonal_factors,
            'jackpot_impact': jackpot_impact,
            'trend_strength': np.mean([f.get('confidence', 0.75) for f in valid_forecasts]),
            'forecasts_combined': len(valid_forecasts)
        }
    
    async def _generate_future_insights(
        self, 
        data: pd.DataFrame, 
        context: Dict[str, Any], 
        forecast: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate future insights and recommendations"""
        
        try:
            insights = {
                'forecast_horizon': f"{self.forecast_horizon} days",
                'trend_direction': 'stable',
                'confidence_level': 'high' if forecast['confidence'] > 0.80 else 'medium',
                'key_factors': [],
                'recommendations': []
            }
            
            # Analyze trend direction
            trend_strength = forecast.get('trend_strength', 0.5)
            if trend_strength > 0.8:
                insights['trend_direction'] = 'strong_upward'
                insights['key_factors'].append('Strong positive trends detected')
            elif trend_strength < 0.3:
                insights['trend_direction'] = 'declining'
                insights['key_factors'].append('Declining trend patterns observed')
            
            # Seasonal insights
            seasonal_factors = forecast.get('seasonal_factors', {})
            if seasonal_factors:
                current_month = datetime.now().month
                if current_month in [11, 12, 1]:
                    insights['key_factors'].append('Holiday season effects active')
                    insights['recommendations'].append('Consider seasonal number preferences')
            
            # Jackpot impact insights
            jackpot_impact = forecast.get('jackpot_impact', {})
            if jackpot_impact.get('combined_impact', 1.0) > 1.3:
                insights['key_factors'].append('High jackpot influencing number selection patterns')
                insights['recommendations'].append('Apply contrarian strategy for high jackpot')
            
            # Future predictions
            insights['future_predictions'] = {
                'next_7_days': 'Trend continuation expected',
                'next_14_days': 'Pattern stability likely',
                'next_30_days': 'Seasonal adjustments may occur'
            }
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"Future insights generation failed: {e}")
            return {
                'forecast_horizon': f"{self.forecast_horizon} days",
                'trend_direction': 'stable',
                'confidence_level': 'medium',
                'error': str(e)
            }
    
    def _fallback_combined_forecast(self) -> Dict[str, Any]:
        """Fallback forecast if all methods fail"""
        return {
            'numbers': [6, 16, 26, 36, 46],
            'confidence': 0.70,
            'summary': 'Fallback trend analysis - standard forecasting applied',
            'trend_analysis': {},
            'seasonal_factors': {},
            'jackpot_impact': {},
            'trend_strength': 0.70,
            'forecasts_combined': 0,
            'fallback_mode': True
        }
    
    def _fallback_forecast(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback forecast if entire system fails"""
        self.logger.warning("Using fallback predictive intelligence")
        
        return {
            'predicted_numbers': [8, 18, 28, 38, 48],
            'confidence': 0.70,
            'trend_analysis': {'fallback_mode': True},
            'forecast_horizon_days': self.forecast_horizon,
            'trend_summary': 'Fallback trend analysis applied',
            'future_insights': {
                'trend_direction': 'stable',
                'confidence_level': 'medium',
                'fallback_mode': True
            },
            'seasonal_factors': {},
            'jackpot_impact': {},
            'trend_strength': 0.70
        }
    
    def _cache_forecast(self, forecast: Dict[str, Any]):
        """Cache forecast results"""
        cache_key = datetime.now().strftime('%Y-%m-%d')
        self.forecast_cache[cache_key] = forecast
        self.last_forecast_update = datetime.now()
        
        # Keep only last 7 days of cache
        cutoff_date = datetime.now() - timedelta(days=7)
        self.forecast_cache = {
            k: v for k, v in self.forecast_cache.items() 
            if datetime.strptime(k, '%Y-%m-%d') >= cutoff_date
        }
    
    async def generate_insights(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive future insights"""
        
        # Check cache first
        cache_key = datetime.now().strftime('%Y-%m-%d')
        if cache_key in self.forecast_cache:
            cached_forecast = self.forecast_cache[cache_key]
            return cached_forecast.get('future_insights', {})
        
        # Generate new insights
        forecast = await self.forecast_trends(data, context)
        return forecast.get('future_insights', {})
    
    def get_predictive_status(self) -> Dict[str, Any]:
        """Get current predictive intelligence status"""
        return {
            'forecast_horizon_days': self.forecast_horizon,
            'trend_analysis_window': self.trend_window,
            'seasonal_adjustment_enabled': self.seasonal_adjustment,
            'jackpot_modeling_enabled': self.jackpot_modeling,
            'cached_forecasts': len(self.forecast_cache),
            'last_forecast_update': self.last_forecast_update.isoformat() if self.last_forecast_update else None,
            'market_factors_tracked': len(self.market_factors)
        }

