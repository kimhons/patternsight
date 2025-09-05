#!/usr/bin/env python3
"""
Universal Lottery Prediction System - Real Data Only Version
Only analyzes lottery types with actual historical data available
"""

import json
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import random
import os

# Lottery configurations for games with real data
LOTTERY_CONFIGS = {
    'megamillions': {
        'name': 'Mega Millions',
        'main_numbers': {'count': 5, 'min': 1, 'max': 70},
        'special_ball': {'count': 1, 'min': 1, 'max': 25, 'name': 'Megaball'},
        'draws_per_week': 2,
        'data_file': 'lottery_data/megamillions.json'
    },
    'luckyforlife': {
        'name': 'Lucky for Life',
        'main_numbers': {'count': 5, 'min': 1, 'max': 48},
        'special_ball': {'count': 1, 'min': 1, 'max': 18, 'name': 'Lucky Ball'},
        'draws_per_week': 2,
        'data_file': 'lottery_data/luckyforlife.json'
    },
    'cash4life': {
        'name': 'Cash4Life',
        'main_numbers': {'count': 5, 'min': 1, 'max': 60},
        'special_ball': {'count': 1, 'min': 1, 'max': 4, 'name': 'Cash Ball'},
        'draws_per_week': 2,
        'data_file': 'lottery_data/cash4life_fixed.json'
    },
    'take5': {
        'name': 'Take 5',
        'main_numbers': {'count': 5, 'min': 1, 'max': 39},
        'special_ball': None,
        'draws_per_week': 14,
        'data_file': 'lottery_data/take5.json'
    },
    'pick10': {
        'name': 'Pick 10',
        'main_numbers': {'count': 10, 'min': 1, 'max': 80},
        'special_ball': None,
        'draws_per_week': 14,
        'data_file': 'lottery_data/pick10.json'
    }
}

class UniversalLotteryAnalyzer:
    """Universal analyzer for lottery types with real data"""
    
    def __init__(self, lottery_type):
        self.lottery_type = lottery_type
        self.config = LOTTERY_CONFIGS.get(lottery_type)
        if not self.config:
            raise ValueError(f"Unknown lottery type: {lottery_type}")
        
        self.draws = []
        self.analysis_results = {}
        
    def load_data(self):
        """Load real lottery data from JSON files"""
        data_file = self.config.get('data_file')
        if not data_file or not os.path.exists(data_file):
            print(f"Warning: No data file available for {self.lottery_type}")
            return False
            
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Check if data contains error response
            if isinstance(data, dict) and data.get('error') == True:
                print(f"Warning: {self.lottery_type} data file contains API error")
                return False
            
            # Parse data based on lottery type
            self.draws = self.parse_lottery_data(data)
            return True
            
        except Exception as e:
            print(f"Error loading data for {self.lottery_type}: {e}")
            return False
    
    def parse_lottery_data(self, data):
        """Parse lottery data from different formats"""
        draws = []
        
        if self.lottery_type == 'megamillions':
            for draw in data:
                if 'winning_numbers' in draw:
                    draws.append({
                        'date': draw.get('draw_date', ''),
                        'numbers': draw['winning_numbers'],
                        'multiplier': draw.get('multiplier')
                    })
        
        elif self.lottery_type == 'luckyforlife':
            for draw in data:
                if 'winning_numbers' in draw:
                    draws.append({
                        'date': draw.get('draw_date', ''),
                        'numbers': draw['winning_numbers'],
                        'lucky_ball': draw.get('lucky_ball')
                    })
        
        elif self.lottery_type == 'cash4life':
            for draw in data:
                if 'winning_numbers' in draw:
                    draws.append({
                        'date': draw.get('draw_date', ''),
                        'numbers': draw['winning_numbers'],
                        'cash_ball': draw.get('cash_ball')
                    })
        
        elif self.lottery_type in ['take5', 'pick10']:
            for draw in data:
                if 'winning_numbers' in draw:
                    draws.append({
                        'date': draw.get('draw_date', ''),
                        'numbers': draw['winning_numbers']
                    })
        
        return draws
    
    def parse_numbers(self, draw):
        """Parse numbers from a draw"""
        numbers_str = draw.get('numbers', '')
        
        # Handle different number formats
        if isinstance(numbers_str, str):
            # Parse space-separated numbers
            parts = numbers_str.split()
            numbers = [int(n) for n in parts if n.isdigit()]
        else:
            numbers = []
        
        # Separate main numbers and special ball
        main_count = self.config['main_numbers']['count']
        main_numbers = numbers[:main_count]
        
        # Get special ball if applicable
        special_number = None
        if self.config['special_ball']:
            if self.lottery_type == 'megamillions':
                # Last number is megaball
                if len(numbers) > main_count:
                    special_number = numbers[main_count]
            elif self.lottery_type == 'luckyforlife':
                special_number = draw.get('lucky_ball')
                if special_number:
                    special_number = int(special_number)
            elif self.lottery_type == 'cash4life':
                special_number = draw.get('cash_ball')
                if special_number:
                    special_number = int(special_number)
        
        return {
            'main': main_numbers,
            'special': special_number
        }
    
    def analyze_frequency(self):
        """Analyze number frequency in real data"""
        main_freq = Counter()
        special_freq = Counter()
        
        for draw in self.draws:
            parsed = self.parse_numbers(draw)
            main_freq.update(parsed['main'])
            if parsed['special']:
                special_freq[parsed['special']] += 1
        
        self.analysis_results['main_frequency'] = dict(main_freq.most_common())
        self.analysis_results['special_frequency'] = dict(special_freq.most_common())
        
        return main_freq, special_freq
    
    def analyze_patterns(self):
        """Analyze patterns in real lottery data"""
        patterns = {
            'consecutive_count': 0,
            'odd_even_distribution': Counter(),
            'sum_statistics': [],
            'range_statistics': [],
            'gap_analysis': defaultdict(int)
        }
        
        last_appearance = {}
        
        for idx, draw in enumerate(self.draws):
            parsed = self.parse_numbers(draw)
            main = parsed['main']
            
            if not main:
                continue
            
            # Track gaps between appearances
            for num in main:
                if num in last_appearance:
                    gap = idx - last_appearance[num]
                    patterns['gap_analysis'][num] += gap
                last_appearance[num] = idx
            
            # Consecutive numbers
            sorted_main = sorted(main)
            for i in range(len(sorted_main) - 1):
                if sorted_main[i+1] - sorted_main[i] == 1:
                    patterns['consecutive_count'] += 1
            
            # Odd/Even distribution
            odd_count = sum(1 for n in main if n % 2 == 1)
            even_count = len(main) - odd_count
            patterns['odd_even_distribution'][(odd_count, even_count)] += 1
            
            # Sum and range statistics
            patterns['sum_statistics'].append(sum(main))
            if len(main) > 1:
                patterns['range_statistics'].append(max(main) - min(main))
        
        self.analysis_results['patterns'] = patterns
        return patterns
    
    def build_markov_model(self, order=1):
        """Build Markov chain model from real data"""
        transitions = defaultdict(Counter)
        
        all_numbers = []
        for draw in self.draws:
            parsed = self.parse_numbers(draw)
            if parsed['main']:
                all_numbers.append(sorted(parsed['main']))
        
        # Build transition matrix
        for i in range(order, len(all_numbers)):
            prev_state = tuple(all_numbers[i-1])
            curr_numbers = all_numbers[i]
            for num in curr_numbers:
                transitions[prev_state][num] += 1
        
        return transitions
    
    def predict_next_draw(self):
        """Generate prediction based on real data analysis"""
        if not self.draws:
            return None
        
        main_freq, special_freq = self.analyze_frequency()
        patterns = self.analyze_patterns()
        markov = self.build_markov_model()
        
        # Calculate prediction scores
        prediction_scores = defaultdict(float)
        
        # 1. Frequency-based scoring (40% weight)
        total_draws = len(self.draws)
        for num, count in main_freq.items():
            prediction_scores[num] += (count / total_draws) * 0.4
        
        # 2. Gap analysis scoring (30% weight)
        if patterns['gap_analysis']:
            max_gap = max(patterns['gap_analysis'].values()) if patterns['gap_analysis'] else 1
            for num, total_gap in patterns['gap_analysis'].items():
                # Numbers with larger gaps are "overdue"
                overdue_score = total_gap / max_gap
                prediction_scores[num] += overdue_score * 0.3
        
        # 3. Markov chain predictions (30% weight)
        if markov and self.draws:
            last_draw = self.parse_numbers(self.draws[-1])
            if last_draw['main']:
                state = tuple(sorted(last_draw['main']))
                
                if state in markov:
                    transitions = markov[state]
                    total_transitions = sum(transitions.values())
                    for num, count in transitions.items():
                        prediction_scores[num] += (count / total_transitions) * 0.3
        
        # Generate final prediction
        main_config = self.config['main_numbers']
        
        # Sort numbers by score
        sorted_nums = sorted(prediction_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top scoring numbers
        prediction = []
        for num, score in sorted_nums:
            if main_config['min'] <= num <= main_config['max'] and num not in prediction:
                prediction.append(num)
                if len(prediction) == main_config['count']:
                    break
        
        # Fill remaining with weighted random selection if needed
        while len(prediction) < main_config['count']:
            available_nums = [n for n in range(main_config['min'], main_config['max'] + 1) 
                            if n not in prediction]
            if available_nums:
                # Weight by frequency
                weights = [main_freq.get(n, 1) for n in available_nums]
                num = random.choices(available_nums, weights=weights)[0]
                prediction.append(num)
        
        prediction.sort()
        
        # Predict special ball if applicable
        special_prediction = None
        if self.config['special_ball'] and special_freq:
            # Use most frequent special number
            special_prediction = special_freq.most_common(1)[0][0]
        
        return {
            'main': prediction,
            'special': special_prediction,
            'confidence': self.calculate_confidence(prediction_scores, prediction)
        }
    
    def calculate_confidence(self, scores, prediction):
        """Calculate confidence based on real data patterns"""
        if not scores or not prediction:
            return 50.0
        
        # Calculate average score of selected numbers
        selected_scores = [scores.get(num, 0) for num in prediction]
        avg_score = np.mean(selected_scores) if selected_scores else 0
        
        # Normalize confidence
        max_possible = max(scores.values()) if scores else 1
        confidence = (avg_score / max_possible) * 100 if max_possible > 0 else 50
        
        # Adjust based on data size
        data_size_factor = min(1.0, len(self.draws) / 1000)
        confidence = confidence * (0.7 + 0.3 * data_size_factor)
        
        return min(95, max(50, confidence))
    
    def generate_report(self):
        """Generate comprehensive analysis report from real data"""
        if not self.draws:
            return None
        
        main_freq, special_freq = self.analyze_frequency()
        patterns = self.analyze_patterns()
        prediction = self.predict_next_draw()
        
        if not prediction:
            return None
        
        report = {
            'lottery_type': self.lottery_type,
            'lottery_name': self.config['name'],
            'total_draws_analyzed': len(self.draws),
            'configuration': {
                'main_numbers': f"{self.config['main_numbers']['count']} numbers from {self.config['main_numbers']['min']} to {self.config['main_numbers']['max']}",
                'special_ball': f"1 from {self.config['special_ball']['min']} to {self.config['special_ball']['max']}" if self.config['special_ball'] else "None",
                'draws_per_week': self.config['draws_per_week']
            },
            'top_10_hot_numbers': dict(Counter(main_freq).most_common(10)),
            'top_10_cold_numbers': dict(sorted(main_freq.items(), key=lambda x: x[1])[:10]),
            'patterns': {
                'consecutive_probability': patterns['consecutive_count'] / len(self.draws) * 100 if self.draws else 0,
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
    """Main execution - analyze lottery types with real data only"""
    
    print("="*80)
    print("UNIVERSAL LOTTERY PREDICTION SYSTEM - REAL DATA ONLY")
    print("Analyzing Lottery Games with Available Historical Data")
    print("="*80)
    
    all_reports = {}
    all_predictions = {}
    
    for lottery_type in LOTTERY_CONFIGS.keys():
        print(f"\n📊 Analyzing {LOTTERY_CONFIGS[lottery_type]['name']}...")
        print("-"*60)
        
        try:
            analyzer = UniversalLotteryAnalyzer(lottery_type)
            
            # Load real data
            if analyzer.load_data():
                report = analyzer.generate_report()
                
                if report:
                    all_reports[lottery_type] = report
                    
                    # Display key findings
                    print(f"✅ Successfully analyzed {report['total_draws_analyzed']} real draws")
                    print(f"📋 Configuration: {report['configuration']['main_numbers']}")
                    
                    if report['configuration']['special_ball'] != "None":
                        print(f"🎱 Special Ball: {report['configuration']['special_ball']}")
                    
                    # Top hot numbers
                    hot_nums = list(report['top_10_hot_numbers'].keys())[:5]
                    print(f"🔥 Hot Numbers: {', '.join(map(str, hot_nums))}")
                    
                    # Top cold numbers  
                    cold_nums = list(report['top_10_cold_numbers'].keys())[:5]
                    print(f"❄️  Cold Numbers: {', '.join(map(str, cold_nums))}")
                    
                    # Display prediction
                    pred = report['prediction']
                    main_str = ', '.join(map(str, pred['main_numbers']))
                    
                    if pred['special_number']:
                        special_name = LOTTERY_CONFIGS[lottery_type]['special_ball']['name']
                        print(f"\n🎯 PREDICTION: [{main_str}] + {special_name}: {pred['special_number']}")
                    else:
                        print(f"\n🎯 PREDICTION: [{main_str}]")
                    
                    print(f"📊 Confidence: {pred['confidence']}")
                    
                    # Store prediction
                    all_predictions[lottery_type] = pred
                else:
                    print(f"⚠️  Unable to generate report for {lottery_type}")
            else:
                print(f"❌ No real data available for {lottery_type}")
                
        except Exception as e:
            print(f"❌ Error analyzing {lottery_type}: {e}")
    
    # Generate comparison report
    if all_reports:
        print("\n" + "="*80)
        print("📈 COMPARATIVE ANALYSIS - REAL DATA ONLY")
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
                comp['Numbers'][:30],  # Truncate if too long
                comp['Draws/Week'],
                comp['Analyzed'],
                comp['Avg Sum'],
                comp['Confidence']
            ))
        
        # Save all reports
        print("\n💾 Saving real data analysis...")
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'system': 'Universal Lottery Prediction System - Real Data v1.0',
            'lotteries_analyzed': len(all_reports),
            'data_source': 'NY Open Data API',
            'reports': all_reports,
            'predictions': all_predictions
        }
        
        with open('universal_lottery_real_analysis.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("✅ Analysis complete! Results saved to universal_lottery_real_analysis.json")
        
        # Display all predictions summary
        print("\n" + "="*80)
        print("🎯 ALL PREDICTIONS SUMMARY - BASED ON REAL DATA")
        print("="*80)
        
        for lottery_type, pred in all_predictions.items():
            name = LOTTERY_CONFIGS[lottery_type]['name']
            main_str = ', '.join(map(str, pred['main_numbers']))
            
            print(f"\n{name}:")
            print(f"  Numbers: [{main_str}]")
            
            if pred['special_number']:
                special_name = LOTTERY_CONFIGS[lottery_type]['special_ball']['name']
                print(f"  {special_name}: {pred['special_number']}")
            
            print(f"  Confidence: {pred['confidence']}")
            print(f"  Data Quality: Real historical data from NY Lottery")
    else:
        print("\n⚠️  No lottery data could be analyzed. Please check data files.")

if __name__ == "__main__":
    main()