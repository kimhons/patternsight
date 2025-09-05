#!/usr/bin/env python3
"""
Universal Lottery Prediction System
Supports multiple lottery types with different configurations
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import random

# Lottery Game Configurations
LOTTERY_CONFIGS = {
    'powerball': {
        'name': 'Powerball',
        'main_numbers': {'count': 5, 'min': 1, 'max': 69},
        'special_ball': {'count': 1, 'min': 1, 'max': 26, 'name': 'Powerball'},
        'draws_per_week': 3
    },
    'megamillions': {
        'name': 'Mega Millions',
        'main_numbers': {'count': 5, 'min': 1, 'max': 70},
        'special_ball': {'count': 1, 'min': 1, 'max': 25, 'name': 'Megaball'},
        'draws_per_week': 2
    },
    'luckyforlife': {
        'name': 'Lucky for Life',
        'main_numbers': {'count': 5, 'min': 1, 'max': 48},
        'special_ball': {'count': 1, 'min': 1, 'max': 18, 'name': 'Lucky Ball'},
        'draws_per_week': 2
    },
    'cash4life': {
        'name': 'Cash4Life',
        'main_numbers': {'count': 5, 'min': 1, 'max': 60},
        'special_ball': {'count': 1, 'min': 1, 'max': 4, 'name': 'Cash Ball'},
        'draws_per_week': 2
    },
    'pick3': {
        'name': 'Pick 3',
        'main_numbers': {'count': 3, 'min': 0, 'max': 9},
        'special_ball': None,
        'draws_per_week': 14,  # Twice daily
        'allow_duplicates': True
    },
    'pick4': {
        'name': 'Pick 4',
        'main_numbers': {'count': 4, 'min': 0, 'max': 9},
        'special_ball': None,
        'draws_per_week': 14,  # Twice daily
        'allow_duplicates': True
    },
    'pick5': {
        'name': 'Pick 5',
        'main_numbers': {'count': 5, 'min': 1, 'max': 39},
        'special_ball': None,
        'draws_per_week': 7
    },
    'take5': {
        'name': 'Take 5',
        'main_numbers': {'count': 5, 'min': 1, 'max': 39},
        'special_ball': None,
        'draws_per_week': 14  # Twice daily
    },
    'pick10': {
        'name': 'Pick 10',
        'main_numbers': {'count': 10, 'min': 1, 'max': 80},
        'special_ball': None,
        'draws_per_week': 14  # Twice daily
    },
    'lottoamerica': {
        'name': 'Lotto America',
        'main_numbers': {'count': 5, 'min': 1, 'max': 52},
        'special_ball': {'count': 1, 'min': 1, 'max': 10, 'name': 'Star Ball'},
        'draws_per_week': 2
    }
}

class UniversalLotteryAnalyzer:
    """Universal analyzer for any lottery type"""
    
    def __init__(self, lottery_type, data=None):
        self.lottery_type = lottery_type
        self.config = LOTTERY_CONFIGS.get(lottery_type)
        if not self.config:
            raise ValueError(f"Unknown lottery type: {lottery_type}")
        
        self.data = data if data else self.load_data()
        self.analysis_results = {}
        
    def load_data(self):
        """Load lottery data from JSON files"""
        file_mapping = {
            'powerball': 'powerball_data_5years.json',
            'megamillions': 'lottery_data/megamillions.json',
            'luckyforlife': 'lottery_data/luckyforlife.json',
            'cash4life': 'lottery_data/cash4life.json',
            'take5': 'lottery_data/take5.json',
            'pick10': 'lottery_data/pick10.json'
        }
        
        filename = file_mapping.get(self.lottery_type)
        if filename:
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                print(f"Data file not found for {self.lottery_type}, generating synthetic data...")
                return self.generate_synthetic_data()
        else:
            return self.generate_synthetic_data()
    
    def generate_synthetic_data(self, num_draws=1000):
        """Generate synthetic lottery data for testing"""
        print(f"Generating synthetic data for {self.config['name']}...")
        
        data = []
        current_date = datetime.now() - timedelta(days=365*5)
        draws_per_week = self.config['draws_per_week']
        days_between_draws = 7.0 / draws_per_week
        
        for i in range(num_draws):
            draw = {
                'draw_date': current_date.strftime('%Y-%m-%d'),
                'winning_numbers': self.generate_random_draw()
            }
            data.append(draw)
            current_date += timedelta(days=days_between_draws)
        
        return data
    
    def generate_random_draw(self):
        """Generate a random draw based on lottery configuration"""
        main_config = self.config['main_numbers']
        
        if self.config.get('allow_duplicates'):
            # For Pick 3/4 where duplicates are allowed
            numbers = [random.randint(main_config['min'], main_config['max']) 
                      for _ in range(main_config['count'])]
        else:
            # For regular lotteries without duplicates
            numbers = sorted(random.sample(
                range(main_config['min'], main_config['max'] + 1),
                main_config['count']
            ))
        
        # Add special ball if exists
        if self.config['special_ball']:
            special = random.randint(
                self.config['special_ball']['min'],
                self.config['special_ball']['max']
            )
            return ' '.join(map(str, numbers + [special]))
        else:
            return ' '.join(map(str, numbers))
    
    def parse_draw(self, draw_string):
        """Parse a draw string into numbers"""
        if isinstance(draw_string, str):
            parts = draw_string.split()
            numbers = [int(n) for n in parts]
        else:
            # Handle different data formats
            if 'winning_numbers' in draw_string:
                return self.parse_draw(draw_string['winning_numbers'])
            return []
        
        main_count = self.config['main_numbers']['count']
        main_numbers = numbers[:main_count]
        special_number = numbers[main_count] if len(numbers) > main_count else None
        
        return {
            'main': main_numbers,
            'special': special_number
        }
    
    def analyze_frequency(self):
        """Analyze number frequency"""
        main_freq = Counter()
        special_freq = Counter()
        
        for draw_data in self.data:
            draw = self.parse_draw(draw_data)
            main_freq.update(draw['main'])
            if draw['special']:
                special_freq[draw['special']] += 1
        
        self.analysis_results['main_frequency'] = dict(main_freq.most_common())
        self.analysis_results['special_frequency'] = dict(special_freq.most_common())
        
        return main_freq, special_freq
    
    def analyze_patterns(self):
        """Analyze patterns specific to lottery type"""
        patterns = {
            'consecutive_count': 0,
            'odd_even_distribution': Counter(),
            'sum_statistics': [],
            'range_statistics': []
        }
        
        for draw_data in self.data:
            draw = self.parse_draw(draw_data)
            main = draw['main']
            
            # Consecutive numbers
            if not self.config.get('allow_duplicates'):
                for i in range(len(main) - 1):
                    if main[i+1] - main[i] == 1:
                        patterns['consecutive_count'] += 1
            
            # Odd/Even
            odd_count = sum(1 for n in main if n % 2 == 1)
            even_count = len(main) - odd_count
            patterns['odd_even_distribution'][(odd_count, even_count)] += 1
            
            # Sum and range
            patterns['sum_statistics'].append(sum(main))
            if len(main) > 1 and not self.config.get('allow_duplicates'):
                patterns['range_statistics'].append(max(main) - min(main))
        
        self.analysis_results['patterns'] = patterns
        return patterns
    
    def build_markov_model(self, order=1):
        """Build Markov chain model for the lottery"""
        transitions = defaultdict(Counter)
        
        draws = [self.parse_draw(d)['main'] for d in self.data]
        
        for i in range(order, len(draws)):
            # For Pick games with duplicates, use different state representation
            if self.config.get('allow_duplicates'):
                prev_state = tuple(draws[i-1])
                curr_numbers = draws[i]
                for num in curr_numbers:
                    transitions[prev_state][num] += 1
            else:
                # For regular lotteries
                prev_state = tuple(sorted(draws[i-1]))
                curr_numbers = draws[i]
                for num in curr_numbers:
                    transitions[prev_state][num] += 1
        
        return transitions
    
    def predict_next_draw(self):
        """Generate prediction based on analysis"""
        main_freq, special_freq = self.analyze_frequency()
        patterns = self.analyze_patterns()
        markov = self.build_markov_model()
        
        # Weight different factors
        prediction_scores = defaultdict(float)
        
        # 1. Frequency-based scoring
        total_draws = len(self.data)
        for num, count in main_freq.items():
            prediction_scores[num] += (count / total_draws) * 0.3
        
        # 2. Pattern-based adjustments
        avg_sum = np.mean(patterns['sum_statistics'])
        std_sum = np.std(patterns['sum_statistics'])
        
        # 3. Markov chain predictions
        if len(markov) > 0:
            recent_draw = self.parse_draw(self.data[-1])['main']
            state = tuple(sorted(recent_draw)) if not self.config.get('allow_duplicates') else tuple(recent_draw)
            
            if state in markov:
                transitions = markov[state]
                total_transitions = sum(transitions.values())
                for num, count in transitions.items():
                    prediction_scores[num] += (count / total_transitions) * 0.4
        
        # Generate final prediction
        main_config = self.config['main_numbers']
        
        if self.config.get('allow_duplicates'):
            # For Pick 3/4/5 with duplicates
            prediction = []
            for _ in range(main_config['count']):
                if prediction_scores:
                    # Weight by scores
                    numbers = list(prediction_scores.keys())
                    weights = list(prediction_scores.values())
                    if sum(weights) > 0:
                        num = random.choices(numbers, weights=weights)[0]
                    else:
                        num = random.randint(main_config['min'], main_config['max'])
                else:
                    num = random.randint(main_config['min'], main_config['max'])
                prediction.append(num)
        else:
            # For regular lotteries without duplicates
            sorted_nums = sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)
            prediction = []
            
            for num, score in sorted_nums:
                if num >= main_config['min'] and num <= main_config['max'] and num not in prediction:
                    prediction.append(num)
                    if len(prediction) == main_config['count']:
                        break
            
            # Fill remaining with random if needed
            while len(prediction) < main_config['count']:
                num = random.randint(main_config['min'], main_config['max'])
                if num not in prediction:
                    prediction.append(num)
            
            prediction.sort()
        
        # Add special ball if needed
        special_prediction = None
        if self.config['special_ball']:
            if special_freq:
                special_prediction = special_freq.most_common(1)[0][0]
            else:
                special_prediction = random.randint(
                    self.config['special_ball']['min'],
                    self.config['special_ball']['max']
                )
        
        return {
            'main': prediction,
            'special': special_prediction,
            'confidence': self.calculate_confidence(prediction_scores, prediction)
        }
    
    def calculate_confidence(self, scores, prediction):
        """Calculate confidence in prediction"""
        if not scores:
            return 50.0
        
        selected_scores = [scores.get(num, 0) for num in prediction]
        avg_score = np.mean(selected_scores) if selected_scores else 0
        max_possible = max(scores.values()) if scores else 1
        
        confidence = (avg_score / max_possible) * 100 if max_possible > 0 else 50
        return min(95, max(50, confidence))
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        main_freq, special_freq = self.analyze_frequency()
        patterns = self.analyze_patterns()
        prediction = self.predict_next_draw()
        
        report = {
            'lottery_type': self.lottery_type,
            'lottery_name': self.config['name'],
            'total_draws_analyzed': len(self.data),
            'configuration': {
                'main_numbers': f"{self.config['main_numbers']['count']} numbers from {self.config['main_numbers']['min']} to {self.config['main_numbers']['max']}",
                'special_ball': f"1 from {self.config['special_ball']['min']} to {self.config['special_ball']['max']}" if self.config['special_ball'] else "None",
                'draws_per_week': self.config['draws_per_week']
            },
            'top_10_hot_numbers': dict(Counter(main_freq).most_common(10)),
            'top_10_cold_numbers': dict(Counter(main_freq).most_common()[-10:]),
            'patterns': {
                'consecutive_probability': patterns['consecutive_count'] / len(self.data) * 100 if not self.config.get('allow_duplicates') else 0,
                'most_common_odd_even': patterns['odd_even_distribution'].most_common(1)[0] if patterns['odd_even_distribution'] else None,
                'average_sum': np.mean(patterns['sum_statistics']) if patterns['sum_statistics'] else 0,
                'std_sum': np.std(patterns['sum_statistics']) if patterns['sum_statistics'] else 0
            },
            'prediction': {
                'main_numbers': prediction['main'],
                'special_number': prediction['special'],
                'confidence': f"{prediction['confidence']:.1f}%"
            }
        }
        
        return report

def main():
    """Main execution - analyze all lottery types"""
    
    print("="*80)
    print("UNIVERSAL LOTTERY PREDICTION SYSTEM")
    print("Analyzing 6 Different Lottery Games")
    print("="*80)
    
    # List of lotteries to analyze
    lotteries = [
        'megamillions',
        'luckyforlife', 
        'cash4life',
        'pick3',
        'pick4',
        'pick5',
        'lottoamerica'
    ]
    
    all_reports = {}
    all_predictions = {}
    
    for lottery_type in lotteries:
        print(f"\n📊 Analyzing {LOTTERY_CONFIGS[lottery_type]['name']}...")
        print("-"*60)
        
        try:
            analyzer = UniversalLotteryAnalyzer(lottery_type)
            report = analyzer.generate_report()
            all_reports[lottery_type] = report
            
            # Display key findings
            print(f"✓ Analyzed {report['total_draws_analyzed']} draws")
            print(f"✓ Configuration: {report['configuration']['main_numbers']}")
            
            if report['configuration']['special_ball'] != "None":
                print(f"✓ Special Ball: {report['configuration']['special_ball']}")
            
            # Display prediction
            pred = report['prediction']
            main_str = ', '.join(map(str, pred['main_numbers']))
            
            if pred['special_number']:
                print(f"\n🎯 PREDICTION: [{main_str}] + Special: {pred['special_number']}")
            else:
                print(f"\n🎯 PREDICTION: [{main_str}]")
            
            print(f"📊 Confidence: {pred['confidence']}")
            
            # Store prediction
            all_predictions[lottery_type] = pred
            
        except Exception as e:
            print(f"❌ Error analyzing {lottery_type}: {e}")
    
    # Generate comparison report
    print("\n" + "="*80)
    print("📈 COMPARATIVE ANALYSIS")
    print("="*80)
    
    comparison = []
    for lottery_type, report in all_reports.items():
        comparison.append({
            'Lottery': report['lottery_name'],
            'Numbers': report['configuration']['main_numbers'],
            'Draws/Week': report['configuration']['draws_per_week'],
            'Analyzed': report['total_draws_analyzed'],
            'Avg Sum': f"{report['patterns']['average_sum']:.1f}",
            'Confidence': report['prediction']['confidence']
        })
    
    # Display comparison table
    print("\n{:<20} {:<30} {:<12} {:<10} {:<10} {:<10}".format(
        'Lottery', 'Configuration', 'Draws/Week', 'Analyzed', 'Avg Sum', 'Confidence'
    ))
    print("-"*100)
    
    for comp in comparison:
        print("{:<20} {:<30} {:<12} {:<10} {:<10} {:<10}".format(
            comp['Lottery'],
            comp['Numbers'],
            comp['Draws/Week'],
            comp['Analyzed'],
            comp['Avg Sum'],
            comp['Confidence']
        ))
    
    # Save all reports
    print("\n💾 Saving comprehensive analysis...")
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'system': 'Universal Lottery Prediction System v1.0',
        'lotteries_analyzed': len(lotteries),
        'reports': all_reports,
        'predictions': all_predictions
    }
    
    with open('universal_lottery_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✅ Analysis complete! Results saved to universal_lottery_analysis.json")
    
    # Display all predictions summary
    print("\n" + "="*80)
    print("🎯 ALL PREDICTIONS SUMMARY")
    print("="*80)
    
    for lottery_type, pred in all_predictions.items():
        name = LOTTERY_CONFIGS[lottery_type]['name']
        main_str = ', '.join(map(str, pred['main_numbers']))
        
        print(f"\n{name}:")
        if pred['special_number']:
            special_name = LOTTERY_CONFIGS[lottery_type]['special_ball']['name']
            print(f"  Numbers: [{main_str}]")
            print(f"  {special_name}: {pred['special_number']}")
        else:
            print(f"  Numbers: [{main_str}]")
        print(f"  Confidence: {pred['confidence']}")

if __name__ == "__main__":
    main()