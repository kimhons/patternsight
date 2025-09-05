#!/usr/bin/env python3
"""
Advanced Markov Chain Model for Powerball Prediction
Enhanced with multi-order chains and transition probabilities
"""

import json
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
import pickle

# Load historical data
with open('powerball_data_5years.json', 'r') as f:
    raw_data = json.load(f)

print("🔗 MARKOV CHAIN ENHANCED PREDICTION SYSTEM")
print("="*80)
print(f"Building Markov models from {len(raw_data)} historical draws...")

class MarkovChainPredictor:
    def __init__(self, order=2):
        """
        Initialize Markov Chain predictor
        order: The number of previous states to consider
        """
        self.order = order
        self.transition_matrix = defaultdict(Counter)
        self.position_chains = [defaultdict(Counter) for _ in range(5)]
        self.powerball_chain = defaultdict(Counter)
        self.sequence_patterns = defaultdict(Counter)
        self.gap_chains = defaultdict(list)
        self.build_models()
    
    def build_models(self):
        """Build multiple Markov chain models from historical data"""
        
        # Process data chronologically
        draws = []
        for entry in sorted(raw_data, key=lambda x: x['draw_date']):
            numbers = [int(n) for n in entry['winning_numbers'].split()]
            main_numbers = sorted(numbers[:5])
            powerball = numbers[5]
            draws.append({
                'main': main_numbers,
                'powerball': powerball,
                'date': entry['draw_date']
            })
        
        print(f"Processing {len(draws)} draws for Markov models...")
        
        # 1. Build transition matrices for consecutive draws
        for i in range(self.order, len(draws)):
            # Get previous states
            prev_states = []
            for j in range(self.order):
                prev_states.append(tuple(draws[i-j-1]['main']))
            
            current = draws[i]['main']
            
            # Overall transition probabilities
            state_key = tuple(prev_states)
            for num in current:
                self.transition_matrix[state_key][num] += 1
            
            # Position-specific chains
            for pos in range(5):
                prev_pos_states = tuple(draws[i-j-1]['main'][pos] for j in range(self.order))
                self.position_chains[pos][prev_pos_states][current[pos]] += 1
            
            # Powerball transitions
            prev_pb_states = tuple(draws[i-j-1]['powerball'] for j in range(self.order))
            self.powerball_chain[prev_pb_states][draws[i]['powerball']] += 1
        
        # 2. Build sequence pattern chains
        for i in range(len(draws) - 1):
            curr_set = set(draws[i]['main'])
            next_set = set(draws[i+1]['main'])
            
            # Numbers that repeat
            repeats = curr_set & next_set
            for num in repeats:
                self.sequence_patterns['repeat'][num] += 1
            
            # Numbers that follow others
            for num1 in curr_set:
                for num2 in next_set:
                    if num1 != num2:
                        self.sequence_patterns[f'follows_{num1}'][num2] += 1
        
        # 3. Build gap-based chains
        number_appearances = defaultdict(list)
        for i, draw in enumerate(draws):
            for num in draw['main']:
                number_appearances[num].append(i)
        
        for num, appearances in number_appearances.items():
            gaps = []
            for i in range(1, len(appearances)):
                gap = appearances[i] - appearances[i-1]
                gaps.append(gap)
            
            if gaps:
                avg_gap = np.mean(gaps)
                std_gap = np.std(gaps) if len(gaps) > 1 else 0
                self.gap_chains[num] = {
                    'avg_gap': avg_gap,
                    'std_gap': std_gap,
                    'last_seen': len(draws) - appearances[-1] - 1,
                    'total_appearances': len(appearances)
                }
        
        print(f"✓ Built {len(self.transition_matrix)} state transitions")
        print(f"✓ Built position-specific chains for 5 positions")
        print(f"✓ Analyzed {len(self.gap_chains)} number gap patterns")
    
    def predict_next_numbers(self, recent_draws=None):
        """Generate prediction based on Markov chains"""
        
        if recent_draws is None:
            # Use last draws from data
            recent_draws = []
            for entry in sorted(raw_data, key=lambda x: x['draw_date'])[-self.order:]:
                numbers = [int(n) for n in entry['winning_numbers'].split()]
                recent_draws.append(sorted(numbers[:5]))
        
        predictions = defaultdict(float)
        
        # 1. Use transition matrix
        if len(recent_draws) >= self.order:
            state_key = tuple(tuple(draw) for draw in recent_draws[-self.order:])
            if state_key in self.transition_matrix:
                transitions = self.transition_matrix[state_key]
                total = sum(transitions.values())
                for num, count in transitions.items():
                    predictions[num] += (count / total) * 0.3  # 30% weight
        
        # 2. Use position-specific predictions
        for pos in range(5):
            if len(recent_draws) >= self.order:
                pos_state = tuple(draw[pos] for draw in recent_draws[-self.order:] if pos < len(draw))
                if pos_state in self.position_chains[pos]:
                    pos_transitions = self.position_chains[pos][pos_state]
                    total = sum(pos_transitions.values())
                    
                    # Boost numbers that fit position
                    for num, count in pos_transitions.most_common(10):
                        predictions[num] += (count / total) * 0.2 * (1 + pos * 0.1)
        
        # 3. Use gap analysis
        for num, gap_info in self.gap_chains.items():
            if gap_info['last_seen'] > 0:
                # Check if number is "due"
                expected_gap = gap_info['avg_gap']
                current_gap = gap_info['last_seen']
                
                if current_gap >= expected_gap:
                    # Number is due or overdue
                    overdue_factor = min(2.0, current_gap / expected_gap)
                    predictions[num] += 0.2 * overdue_factor
                else:
                    # Number appeared recently, less likely
                    predictions[num] += 0.05
        
        # 4. Use sequence patterns
        if recent_draws:
            last_draw = set(recent_draws[-1])
            
            # Check for numbers that tend to repeat
            for num in last_draw:
                if num in self.sequence_patterns['repeat']:
                    repeat_prob = self.sequence_patterns['repeat'][num] / len(raw_data)
                    predictions[num] += 0.1 * repeat_prob
            
            # Check for following patterns
            for num in last_draw:
                follow_key = f'follows_{num}'
                if follow_key in self.sequence_patterns:
                    followers = self.sequence_patterns[follow_key]
                    total = sum(followers.values())
                    for follow_num, count in followers.most_common(5):
                        predictions[follow_num] += (count / total) * 0.2
        
        # Select top predictions
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure we have valid numbers
        final_numbers = []
        for num, score in sorted_predictions:
            if 1 <= num <= 69 and num not in final_numbers:
                final_numbers.append(num)
                if len(final_numbers) == 5:
                    break
        
        # Fill if needed
        while len(final_numbers) < 5:
            num = np.random.randint(1, 70)
            if num not in final_numbers:
                final_numbers.append(num)
        
        return sorted(final_numbers)
    
    def predict_powerball(self, recent_powerballs=None):
        """Predict next Powerball using Markov chain"""
        
        if recent_powerballs is None:
            # Get last powerballs
            recent_powerballs = []
            for entry in sorted(raw_data, key=lambda x: x['draw_date'])[-self.order:]:
                numbers = [int(n) for n in entry['winning_numbers'].split()]
                recent_powerballs.append(numbers[5])
        
        if len(recent_powerballs) >= self.order:
            state_key = tuple(recent_powerballs[-self.order:])
            if state_key in self.powerball_chain:
                transitions = self.powerball_chain[state_key]
                # Return most likely
                return transitions.most_common(1)[0][0]
        
        # Fallback to most common powerball
        all_pb = []
        for entry in raw_data:
            numbers = [int(n) for n in entry['winning_numbers'].split()]
            all_pb.append(numbers[5])
        
        pb_freq = Counter(all_pb)
        return pb_freq.most_common(1)[0][0]
    
    def analyze_patterns(self):
        """Analyze and display Markov chain patterns"""
        
        print("\n📊 MARKOV CHAIN ANALYSIS")
        print("="*80)
        
        # Most common state transitions
        print("\n🔄 Top State Transitions:")
        top_transitions = sorted(
            [(state, num, count) for state, nums in self.transition_matrix.items() 
             for num, count in nums.items()],
            key=lambda x: x[2],
            reverse=True
        )[:10]
        
        for state, num, count in top_transitions:
            recent_nums = state[0] if state else []
            print(f"  After {recent_nums[:5]} → {num} ({count} times)")
        
        # Position-specific patterns
        print("\n📍 Position-Specific Markov Patterns:")
        for pos in range(5):
            top_pos = sorted(
                [(state, num, count) for state, nums in self.position_chains[pos].items()
                 for num, count in nums.items()],
                key=lambda x: x[2],
                reverse=True
            )[:3]
            
            print(f"\n  Position {pos + 1}:")
            for state, num, count in top_pos:
                print(f"    After {state} → {num} ({count} times)")
        
        # Gap patterns
        print("\n⏱️ Gap-Based Predictions (Numbers Due):")
        overdue = sorted(
            [(num, info['last_seen'], info['avg_gap']) 
             for num, info in self.gap_chains.items()
             if info['last_seen'] > info['avg_gap']],
            key=lambda x: x[1] / x[2],
            reverse=True
        )[:10]
        
        for num, last_seen, avg_gap in overdue:
            overdue_factor = last_seen / avg_gap
            print(f"  Number {num:2d}: Last seen {last_seen:3d} draws ago (avg: {avg_gap:.1f}) - {overdue_factor:.1f}x overdue")
        
        # Sequence patterns
        print("\n🔁 Numbers That Tend to Repeat:")
        repeat_probs = sorted(
            [(num, count / len(raw_data) * 100) 
             for num, count in self.sequence_patterns['repeat'].items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for num, prob in repeat_probs:
            print(f"  Number {num:2d}: {prob:.1f}% repeat probability")

class HierarchicalMarkovChain:
    """Multi-level Markov chain with different granularities"""
    
    def __init__(self):
        self.models = {
            'order_1': MarkovChainPredictor(order=1),
            'order_2': MarkovChainPredictor(order=2),
            'order_3': MarkovChainPredictor(order=3)
        }
        self.weights = {'order_1': 0.2, 'order_2': 0.5, 'order_3': 0.3}
    
    def predict(self):
        """Combine predictions from multiple order Markov chains"""
        
        all_predictions = defaultdict(float)
        
        for name, model in self.models.items():
            prediction = model.predict_next_numbers()
            weight = self.weights[name]
            
            for num in prediction:
                all_predictions[num] += weight
        
        # Get top 5
        sorted_nums = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
        final = [num for num, _ in sorted_nums[:5]]
        
        # Get powerball from best model
        powerball = self.models['order_2'].predict_powerball()
        
        return sorted(final), powerball

# Main execution
def main():
    # Create models
    print("\n🔨 Building Markov Chain Models...")
    
    # Single-order models
    model_1 = MarkovChainPredictor(order=1)
    model_2 = MarkovChainPredictor(order=2)
    model_3 = MarkovChainPredictor(order=3)
    
    # Hierarchical model
    hierarchical = HierarchicalMarkovChain()
    
    # Analyze patterns
    model_2.analyze_patterns()
    
    # Generate predictions
    print("\n" + "="*80)
    print("🎯 MARKOV CHAIN PREDICTIONS")
    print("="*80)
    
    predictions = []
    
    # Generate 10 predictions
    for i in range(10):
        # Use different models
        if i < 3:
            nums = model_1.predict_next_numbers()
            pb = model_1.predict_powerball()
            method = "1st-order"
        elif i < 6:
            nums = model_2.predict_next_numbers()
            pb = model_2.predict_powerball()
            method = "2nd-order"
        elif i < 8:
            nums = model_3.predict_next_numbers()
            pb = model_3.predict_powerball()
            method = "3rd-order"
        else:
            nums, pb = hierarchical.predict()
            method = "Hierarchical"
        
        predictions.append({
            'id': i + 1,
            'numbers': nums,
            'powerball': pb,
            'method': method
        })
        
        print(f"{i+1:2d}. [{', '.join(f'{n:2d}' for n in nums)}] + PB: {pb:2d} ({method})")
    
    # Statistical analysis
    print("\n📊 PREDICTION ANALYSIS")
    print("="*80)
    
    all_nums = []
    for pred in predictions:
        all_nums.extend(pred['numbers'])
    
    num_freq = Counter(all_nums)
    top_nums = num_freq.most_common(10)
    
    print("\nMost Selected Numbers:")
    for num, count in top_nums:
        pct = count / 10 * 100 / 5
        bar = '█' * count
        print(f"  {num:2d}: {count}x ({pct:5.1f}%) {bar}")
    
    # Save model
    print("\n💾 Saving Markov models...")
    
    model_data = {
        'order_2_transitions': dict(model_2.transition_matrix),
        'position_chains': [dict(pc) for pc in model_2.position_chains],
        'powerball_chain': dict(model_2.powerball_chain),
        'gap_chains': dict(model_2.gap_chains),
        'predictions': predictions
    }
    
    with open('markov_model.json', 'w') as f:
        json.dump(model_data, f, default=str, indent=2)
    
    print("✅ Markov models saved to markov_model.json")
    
    # Final recommendation
    print("\n" + "="*80)
    print("🎯 MARKOV CHAIN RECOMMENDATION")
    print("="*80)
    
    # Use 2nd-order as primary
    final_nums = model_2.predict_next_numbers()
    final_pb = model_2.predict_powerball()
    
    print(f"\nBest Markov Prediction: [{', '.join(f'{n:2d}' for n in final_nums)}] + Powerball: {final_pb}")
    print("\nBased on:")
    print(f"  • {len(raw_data)} historical draws")
    print(f"  • 2nd-order Markov chain analysis")
    print(f"  • Position-specific transition probabilities")
    print(f"  • Gap analysis for overdue numbers")
    print(f"  • Sequence pattern recognition")
    
    return predictions

if __name__ == "__main__":
    predictions = main()
    print("\n✅ Markov Chain Enhancement Complete!")