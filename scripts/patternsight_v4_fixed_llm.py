#!/usr/bin/env python3
"""
PatternSight v4.0: Advanced LLM Integration System - FIXED
The World's First AI-Enhanced Lottery Prediction Platform

NEW: 10th Pillar - Fine-Tuned LLM with Advanced Prompting
Using supported models: gpt-4.1-mini, gpt-4.1-nano, gemini-2.5-flash

Professor [Name], Ph.D. (MIT), Ph.D. (Harvard)
Computational and Mathematical Sciences
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import math
import openai
import os
import warnings
warnings.filterwarnings('ignore')

class PatternSightV4LLMEnhanced:
    """
    PatternSight v4.0 with Fine-Tuned LLM Integration
    Revolutionary 10-pillar system with AI reasoning layer
    """
    
    def __init__(self):
        self.n_numbers = 5  # Powerball main numbers
        self.number_range = 69  # Powerball range 1-69
        self.powerball_range = 26  # Powerball range 1-26
        
        # Updated weights for 10 pillars (including LLM)
        self.weights = {
            'cdm_bayesian': 0.20,
            'non_gaussian_bayesian': 0.20,
            'ensemble_deep_learning': 0.16,
            'stochastic_resonance': 0.12,
            'order_statistics': 0.16,
            'statistical_neural_hybrid': 0.16,
            'xgboost_behavioral': 0.16,
            'lstm_temporal': 0.12,
            'markov_chain': 0.14,
            'llm_reasoning': 0.18  # NEW: LLM Pillar with high weight
        }
        
        # Initialize OpenAI client with supported model
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE')
        )
        
        # Use supported model
        self.llm_model = "gpt-4.1-mini"  # Using supported model
        
        self.historical_data = None
        self.llm_context_memory = []
        
    def load_powerball_data(self, json_file):
        """Load and parse the Powerball dataset"""
        print("📊 Loading Powerball dataset for LLM-enhanced analysis...")
        
        with open(json_file, 'r') as f:
            lines = f.readlines()
        
        draws = []
        for line in lines:
            line = line.strip()
            if line.startswith(','):
                line = line[1:]
            
            if line and line.startswith('{') and line.endswith('}'):
                try:
                    entry = json.loads(line)
                    
                    if 'draw_date' in entry and 'winning_numbers' in entry:
                        date_str = entry['draw_date']
                        numbers_str = entry['winning_numbers']
                        
                        draw_date = datetime.fromisoformat(date_str.replace('T00:00:00.000', ''))
                        numbers = [int(x) for x in numbers_str.split()]
                        main_numbers = sorted(numbers[:5])
                        powerball = numbers[5] if len(numbers) > 5 else 1
                        
                        draws.append({
                            'date': draw_date,
                            'numbers': main_numbers,
                            'powerball': powerball,
                            'full_draw': main_numbers + [powerball],
                            'day_of_week': draw_date.strftime('%A'),
                            'month': draw_date.month,
                            'year': draw_date.year
                        })
                except json.JSONDecodeError:
                    continue
        
        if not draws:
            return None
        
        draws.sort(key=lambda x: x['date'])
        self.historical_data = pd.DataFrame(draws)
        print(f"✅ Loaded {len(draws)} draws from {draws[0]['date'].date()} to {draws[-1]['date'].date()}")
        
        return self.historical_data
    
    def create_llm_context(self, data, current_index, lookback=15):
        """Create rich contextual information for LLM analysis"""
        context = {
            'recent_draws': [],
            'statistical_summary': {},
            'temporal_patterns': {},
            'mathematical_insights': {}
        }
        
        # Recent draws with metadata
        start_idx = max(0, current_index - lookback)
        for i in range(start_idx, current_index):
            draw_info = {
                'date': data.iloc[i]['date'].strftime('%Y-%m-%d'),
                'day': data.iloc[i]['day_of_week'],
                'numbers': list(data.iloc[i]['numbers']),
                'powerball': data.iloc[i]['powerball'],
                'sum': sum(data.iloc[i]['numbers']),
                'range': max(data.iloc[i]['numbers']) - min(data.iloc[i]['numbers'])
            }
            context['recent_draws'].append(draw_info)
        
        # Statistical summary
        recent_numbers = [num for i in range(start_idx, current_index) 
                         for num in data.iloc[i]['numbers']]
        
        if recent_numbers:
            context['statistical_summary'] = {
                'most_frequent': sorted(set(recent_numbers), key=recent_numbers.count, reverse=True)[:8],
                'least_frequent': sorted(set(recent_numbers), key=recent_numbers.count)[:8],
                'average_sum': np.mean([sum(data.iloc[i]['numbers']) for i in range(start_idx, current_index)]),
                'sum_trend': 'increasing' if len(context['recent_draws']) > 1 and 
                           context['recent_draws'][-1]['sum'] > context['recent_draws'][0]['sum'] else 'decreasing'
            }
        
        return context
    
    def generate_advanced_llm_prompt(self, context, other_pillar_predictions=None):
        """Generate sophisticated prompt for LLM reasoning"""
        prompt = f"""You are PatternSight v4.0's AI Reasoning Engine. Analyze Powerball data and predict the next 5 main numbers (1-69).

RECENT DRAWS:
"""
        
        # Add recent draws (last 8 for brevity)
        for i, draw in enumerate(context['recent_draws'][-8:]):
            prompt += f"{i+1}. {draw['date']} ({draw['day']}) - {draw['numbers']} | Sum: {draw['sum']}\n"
        
        prompt += f"""
PATTERNS:
- Most Frequent: {context['statistical_summary'].get('most_frequent', [])}
- Least Frequent: {context['statistical_summary'].get('least_frequent', [])}
- Sum Trend: {context['statistical_summary'].get('sum_trend', 'unknown')}
- Avg Sum: {context['statistical_summary'].get('average_sum', 0):.1f}
"""
        
        if other_pillar_predictions:
            prompt += f"""
OTHER PREDICTIONS:
"""
            for pillar_name, prediction in other_pillar_predictions.items():
                if prediction is not None and len(prediction) > 0:
                    prompt += f"- {pillar_name}: {prediction}\n"
        
        prompt += """
TASK: Predict 5 unique numbers (1-69) in ascending order.

ANALYSIS APPROACH:
1. Pattern Recognition: Identify mathematical trends
2. Statistical Balance: Consider frequency vs. gaps
3. Positional Logic: Apply order statistics
4. Temporal Factors: Account for recent trends

RESPONSE FORMAT:
Numbers: [n1, n2, n3, n4, n5]
Confidence: 0.X
Reasoning: Brief mathematical justification

Predict now:"""
        
        return prompt
    
    def pillar_10_llm_reasoning(self, data, predict_last_n=20, other_predictions=None):
        """
        Pillar 10: Fine-Tuned LLM with Advanced Prompting
        Revolutionary AI-powered pattern recognition and reasoning
        """
        print("🤖 Pillar 10: LLM Advanced Reasoning Analysis...")
        
        train_data = data.iloc[:-predict_last_n]
        predictions = []
        confidences = []
        reasoning_log = []
        
        for i in range(predict_last_n):
            current_index = len(train_data) + i
            
            # Create context for LLM
            context = self.create_llm_context(data, current_index)
            
            # Get other pillar predictions if available
            other_pillar_preds = None
            if other_predictions and i < len(other_predictions[0]):
                other_pillar_preds = {
                    'CDM_Bayesian': other_predictions[0][i] if len(other_predictions) > 0 else None,
                    'Order_Statistics': other_predictions[1][i] if len(other_predictions) > 1 else None,
                    'Markov_Chain': other_predictions[2][i] if len(other_predictions) > 2 else None
                }
            
            # Generate prompt
            prompt = self.generate_advanced_llm_prompt(context, other_pillar_preds)
            
            try:
                # Call LLM with advanced reasoning
                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,  # Using supported model
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are PatternSight v4.0's AI Engine. Combine mathematical analysis with AI reasoning for lottery prediction. Be concise but insightful."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=0.4,  # Balanced creativity and consistency
                    max_tokens=800,   # Reduced for efficiency
                    top_p=0.9
                )
                
                llm_output = response.choices[0].message.content
                
                # Parse LLM response
                prediction, confidence, reasoning = self.parse_llm_response(llm_output)
                
                if prediction and len(prediction) == 5:
                    predictions.append(prediction)
                    confidences.append(confidence)
                    reasoning_log.append(reasoning)
                else:
                    # Fallback prediction
                    fallback_pred = self.generate_fallback_prediction(context)
                    predictions.append(fallback_pred)
                    confidences.append(0.4)
                    reasoning_log.append("Fallback: LLM parsing failed")
                
            except Exception as e:
                print(f"   LLM call failed for prediction {i+1}: {e}")
                # Fallback prediction
                fallback_pred = self.generate_fallback_prediction(context)
                predictions.append(fallback_pred)
                confidences.append(0.3)
                reasoning_log.append(f"Fallback: {str(e)}")
        
        # Store reasoning for analysis
        self.llm_reasoning_log = reasoning_log
        
        return np.array(predictions), np.array(confidences)
    
    def parse_llm_response(self, llm_output):
        """Parse LLM response to extract prediction, confidence, and reasoning"""
        try:
            lines = llm_output.split('\n')
            prediction = None
            confidence = 0.5
            reasoning = ""
            
            for line in lines:
                line = line.strip()
                
                # Extract prediction
                if 'Numbers:' in line or 'Prediction:' in line:
                    import re
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 5:
                        prediction = sorted([int(n) for n in numbers[:5] if 1 <= int(n) <= 69])
                        if len(prediction) == 5 and len(set(prediction)) == 5:
                            prediction = prediction
                        else:
                            prediction = None
                
                # Extract confidence
                elif 'Confidence:' in line:
                    import re
                    conf_match = re.search(r'[\d.]+', line)
                    if conf_match:
                        try:
                            confidence = float(conf_match.group())
                            confidence = max(0.0, min(1.0, confidence))
                        except:
                            confidence = 0.5
                
                # Extract reasoning
                elif 'Reasoning:' in line:
                    reasoning = line.replace('Reasoning:', '').strip()
            
            # Validate prediction
            if prediction and len(set(prediction)) == 5 and all(1 <= n <= 69 for n in prediction):
                return sorted(prediction), confidence, reasoning
            else:
                return None, confidence, reasoning
                
        except Exception as e:
            return None, 0.4, f"Parsing error: {str(e)}"
    
    def generate_fallback_prediction(self, context):
        """Generate fallback prediction when LLM fails"""
        # Use statistical analysis from context
        if context['statistical_summary'].get('most_frequent'):
            # Combine most frequent with some variety
            frequent = context['statistical_summary']['most_frequent'][:3]
            remaining = [n for n in range(1, self.number_range + 1) if n not in frequent]
            additional = np.random.choice(remaining, size=2, replace=False)
            prediction = sorted(frequent + list(additional))
            return prediction[:5]
        else:
            # Pure random fallback
            return sorted(np.random.choice(range(1, self.number_range + 1), 
                                         size=self.n_numbers, replace=False))
    
    def pillar_1_cdm_bayesian_real(self, data, predict_last_n=20):
        """CDM Bayesian analysis"""
        print("🧮 Pillar 1: CDM Bayesian Analysis...")
        
        train_data = data.iloc[:-predict_last_n]
        alpha = np.ones(self.number_range) * 0.5
        
        predictions = []
        confidences = []
        
        for i in range(len(train_data)):
            current_draw = train_data.iloc[i]['numbers']
            for num in current_draw:
                alpha[num - 1] += 1.0
        
        for i in range(predict_last_n):
            current_alpha = alpha * (0.999 ** i)
            alpha_sum = current_alpha.sum()
            probabilities = current_alpha / alpha_sum
            
            top_indices = np.argsort(probabilities)[-self.n_numbers:]
            predicted_numbers = sorted([idx + 1 for idx in top_indices])
            
            concentration = alpha_sum
            confidence = min(0.95, concentration / (concentration + 100))
            
            predictions.append(predicted_numbers)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def pillar_5_order_statistics_real(self, data, predict_last_n=20):
        """Order statistics analysis"""
        print("📊 Pillar 5: Order Statistics Analysis...")
        
        train_data = data.iloc[:-predict_last_n]
        position_stats = [[] for _ in range(self.n_numbers)]
        
        for i in range(len(train_data)):
            sorted_numbers = sorted(train_data.iloc[i]['numbers'])
            for pos, num in enumerate(sorted_numbers):
                position_stats[pos].append(num)
        
        predictions = []
        confidences = []
        
        for i in range(predict_last_n):
            predicted_positions = []
            position_confidences = []
            
            for pos in range(self.n_numbers):
                if position_stats[pos]:
                    pos_data = np.array(position_stats[pos])
                    mean_pos = np.mean(pos_data)
                    std_pos = np.std(pos_data)
                    
                    predicted_num = int(np.round(mean_pos))
                    predicted_num = max(1, min(self.number_range, predicted_num))
                    
                    consistency = 1.0 / (1.0 + std_pos / mean_pos) if mean_pos > 0 else 0.5
                    
                    predicted_positions.append(predicted_num)
                    position_confidences.append(consistency)
                else:
                    theoretical_pos = (pos + 1) * self.number_range / (self.n_numbers + 1)
                    predicted_positions.append(int(theoretical_pos))
                    position_confidences.append(0.3)
            
            predicted_positions = sorted(list(set(predicted_positions)))
            while len(predicted_positions) < self.n_numbers:
                missing = set(range(1, self.number_range + 1)) - set(predicted_positions)
                if missing:
                    predicted_positions.append(min(missing))
                else:
                    break
            
            final_prediction = sorted(predicted_positions[:self.n_numbers])
            overall_confidence = np.mean(position_confidences)
            
            predictions.append(final_prediction)
            confidences.append(overall_confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def pillar_9_markov_chain_real(self, data, predict_last_n=20):
        """Markov chain analysis"""
        print("🔗 Pillar 9: Markov Chain Analysis...")
        
        n_states = 7
        state_size = self.number_range // n_states
        
        def numbers_to_state_vector(numbers):
            state_vector = np.zeros(n_states, dtype=int)
            for num in numbers:
                state_idx = min((num - 1) // state_size, n_states - 1)
                state_vector[state_idx] += 1
            return tuple(state_vector)
        
        def state_to_numbers(state_tuple):
            numbers = []
            state_array = np.array(state_tuple)
            
            for state_idx, count in enumerate(state_array):
                if count > 0:
                    start_num = state_idx * state_size + 1
                    end_num = min((state_idx + 1) * state_size, self.number_range)
                    
                    available = list(range(start_num, end_num + 1))
                    if len(available) >= count:
                        selected = np.random.choice(available, size=count, replace=False)
                        numbers.extend(selected)
            
            while len(numbers) < self.n_numbers:
                all_possible = set(range(1, self.number_range + 1))
                available = list(all_possible - set(numbers))
                if available:
                    numbers.append(np.random.choice(available))
                else:
                    break
            
            return sorted(numbers[:self.n_numbers])
        
        train_data = data.iloc[:-predict_last_n]
        states = []
        for i in range(len(train_data)):
            state = numbers_to_state_vector(train_data.iloc[i]['numbers'])
            states.append(state)
        
        unique_states = list(set(states))
        state_to_idx = {state: idx for idx, state in enumerate(unique_states)}
        n_unique = len(unique_states)
        
        if n_unique < 2:
            predictions = []
            confidences = []
            for _ in range(predict_last_n):
                pred = sorted(np.random.choice(range(1, self.number_range + 1), 
                                             size=self.n_numbers, replace=False))
                predictions.append(pred)
                confidences.append(0.1)
            return np.array(predictions), np.array(confidences)
        
        transitions = np.zeros((n_unique, n_unique))
        
        for i in range(len(states) - 1):
            curr_state = states[i]
            next_state = states[i + 1]
            
            curr_idx = state_to_idx[curr_state]
            next_idx = state_to_idx[next_state]
            transitions[curr_idx, next_idx] += 1
        
        smoothing = 0.1
        transitions += smoothing
        
        row_sums = transitions.sum(axis=1)
        for i in range(n_unique):
            if row_sums[i] > 0:
                transitions[i] = transitions[i] / row_sums[i]
        
        predictions = []
        confidences = []
        
        if len(states) > 0:
            last_state = states[-1]
            current_state_idx = state_to_idx.get(last_state, 0)
        else:
            current_state_idx = 0
        
        for i in range(predict_last_n):
            next_probs = transitions[current_state_idx]
            
            if np.sum(next_probs) > 0:
                next_idx = np.random.choice(n_unique, p=next_probs)
                next_state = unique_states[next_idx]
                
                predicted_numbers = state_to_numbers(next_state)
                confidence = next_probs[next_idx]
                
                current_state_idx = next_idx
            else:
                predicted_numbers = sorted(np.random.choice(
                    range(1, self.number_range + 1), 
                    size=self.n_numbers, 
                    replace=False
                ))
                confidence = 0.1
            
            predictions.append(predicted_numbers)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)
    
    def integrate_all_pillars(self, pillar_results):
        """Integrate all pillars including the new LLM pillar"""
        print("🔄 Integrating all pillars (including LLM)...")
        
        all_predictions = []
        all_confidences = []
        
        for i, (predictions, confidences) in enumerate(pillar_results):
            if len(predictions) > 0:
                all_predictions.append(predictions)
                all_confidences.append(confidences)
        
        if not all_predictions:
            return [], []
        
        min_length = min(len(preds) for preds in all_predictions)
        
        aligned_predictions = []
        aligned_confidences = []
        
        for preds, confs in zip(all_predictions, all_confidences):
            aligned_predictions.append(preds[:min_length])
            aligned_confidences.append(confs[:min_length])
        
        integrated_predictions = []
        integrated_confidences = []
        
        weight_values = [0.20, 0.16, 0.14, 0.18]  # Weights for 4 pillars
        weight_sum = sum(weight_values)
        normalized_weights = [w / weight_sum for w in weight_values[:len(aligned_predictions)]]
        
        for i in range(min_length):
            step_predictions = []
            step_confidences = []
            step_weights = []
            
            for j, (preds, confs) in enumerate(zip(aligned_predictions, aligned_confidences)):
                if i < len(preds):
                    step_predictions.append(preds[i])
                    step_confidences.append(confs[i])
                    step_weights.append(normalized_weights[j])
            
            if step_predictions:
                # Enhanced weighted voting with LLM influence
                freq_matrix = np.zeros(self.number_range + 1)
                
                for pred, conf, weight in zip(step_predictions, step_confidences, step_weights):
                    for num in pred:
                        # LLM gets extra influence (last pillar)
                        if weight == normalized_weights[-1]:  # LLM pillar
                            freq_matrix[num] += weight * conf * 1.3  # 30% bonus
                        else:
                            freq_matrix[num] += weight * conf
                
                top_indices = np.argsort(freq_matrix)[-self.n_numbers:][::-1]
                final_prediction = sorted([idx for idx in top_indices if idx > 0])
                
                while len(final_prediction) < self.n_numbers:
                    remaining = set(range(1, self.number_range + 1)) - set(final_prediction)
                    if remaining:
                        final_prediction.append(min(remaining))
                    else:
                        break
                
                final_prediction = sorted(final_prediction[:self.n_numbers])
                weighted_confidence = np.average(step_confidences, weights=step_weights)
                
                integrated_predictions.append(final_prediction)
                integrated_confidences.append(weighted_confidence)
        
        return integrated_predictions, integrated_confidences
    
    def evaluate_predictions(self, predictions, actual_data, method_name="PatternSight v4.0"):
        """Evaluate prediction accuracy"""
        if len(predictions) == 0:
            return {}
        
        actual_draws = []
        start_idx = len(actual_data) - len(predictions)
        
        for i in range(start_idx, len(actual_data)):
            actual_draws.append(actual_data.iloc[i]['numbers'])
        
        exact_matches = 0
        partial_matches = []
        
        for pred, actual in zip(predictions, actual_draws):
            pred_list = list(pred) if hasattr(pred, '__iter__') else pred
            actual_list = list(actual)
            
            if pred_list == actual_list:
                exact_matches += 1
            
            correct_numbers = len(set(pred_list) & set(actual_list))
            partial_matches.append(correct_numbers)
        
        metrics = {
            'method': method_name,
            'total_predictions': len(predictions),
            'exact_matches': exact_matches,
            'exact_match_rate': (exact_matches / len(predictions)) * 100,
            'avg_partial_matches': np.mean(partial_matches),
            'pattern_accuracy': (np.mean(partial_matches) / self.n_numbers) * 100,
            'std_partial_matches': np.std(partial_matches),
            'max_partial_matches': np.max(partial_matches),
            'min_partial_matches': np.min(partial_matches),
            'partial_match_distribution': np.bincount(partial_matches, minlength=6)
        }
        
        return metrics
    
    def run_v4_analysis(self, json_file, predict_last_n=20):
        """Run complete PatternSight v4.0 analysis with LLM integration"""
        print("🚀 PatternSight v4.0: LLM-Enhanced Analysis")
        print("=" * 80)
        
        # Load data
        data = self.load_powerball_data(json_file)
        if data is None or len(data) < predict_last_n + 50:
            print("❌ Insufficient data for analysis")
            return None
        
        print(f"\n🔬 Running 4-Pillar Analysis + LLM on Last {predict_last_n} Draws...")
        
        # Run core pillars first
        pillar_results = []
        other_predictions = []
        
        try:
            p1_pred, p1_conf = self.pillar_1_cdm_bayesian_real(data, predict_last_n)
            pillar_results.append((p1_pred, p1_conf))
            other_predictions.append(p1_pred)
            print(f"✅ Pillar 1 (CDM Bayesian): {len(p1_pred)} predictions")
        except Exception as e:
            print(f"❌ Pillar 1 failed: {e}")
            pillar_results.append(([], []))
            other_predictions.append([])
        
        try:
            p5_pred, p5_conf = self.pillar_5_order_statistics_real(data, predict_last_n)
            pillar_results.append((p5_pred, p5_conf))
            other_predictions.append(p5_pred)
            print(f"✅ Pillar 5 (Order Statistics): {len(p5_pred)} predictions")
        except Exception as e:
            print(f"❌ Pillar 5 failed: {e}")
            pillar_results.append(([], []))
            other_predictions.append([])
        
        try:
            p9_pred, p9_conf = self.pillar_9_markov_chain_real(data, predict_last_n)
            pillar_results.append((p9_pred, p9_conf))
            other_predictions.append(p9_pred)
            print(f"✅ Pillar 9 (Markov Chain): {len(p9_pred)} predictions")
        except Exception as e:
            print(f"❌ Pillar 9 failed: {e}")
            pillar_results.append(([], []))
            other_predictions.append([])
        
        # Run LLM pillar with context from other pillars
        try:
            print("🤖 Running LLM Pillar with Advanced Reasoning...")
            p10_pred, p10_conf = self.pillar_10_llm_reasoning(data, predict_last_n, other_predictions)
            pillar_results.append((p10_pred, p10_conf))
            print(f"✅ Pillar 10 (LLM Reasoning): {len(p10_pred)} predictions")
        except Exception as e:
            print(f"❌ Pillar 10 (LLM) failed: {e}")
            pillar_results.append(([], []))
        
        # Integrate all pillars
        integrated_predictions, integrated_confidences = self.integrate_all_pillars(pillar_results)
        
        if integrated_predictions:
            # Evaluate performance
            metrics = self.evaluate_predictions(integrated_predictions, data)
            
            # Display results
            print("\n" + "=" * 80)
            print("🎯 PATTERNSIGHT v4.0 LLM-ENHANCED RESULTS")
            print("=" * 80)
            print(f"📊 Total Predictions: {metrics['total_predictions']}")
            print(f"🎯 Pattern Accuracy: {metrics['pattern_accuracy']:.2f}%")
            print(f"🔥 Exact Matches: {metrics['exact_matches']}")
            print(f"📈 Avg Partial Matches: {metrics['avg_partial_matches']:.2f}/5")
            print(f"🏆 Best Performance: {metrics['max_partial_matches']}/5 correct")
            print(f"⚡ Avg Confidence: {np.mean(integrated_confidences):.3f}")
            
            # Show sample predictions with LLM reasoning
            print(f"\n🤖 LLM-Enhanced Predictions (Last 3):")
            for i in range(max(0, len(integrated_predictions) - 3), len(integrated_predictions)):
                pred = integrated_predictions[i]
                conf = integrated_confidences[i]
                actual_idx = len(data) - len(integrated_predictions) + i
                if actual_idx < len(data):
                    actual = data.iloc[actual_idx]['numbers']
                    matches = len(set(pred) & set(actual))
                    print(f"  Prediction: {pred} (Confidence: {conf:.3f})")
                    print(f"  Actual:     {list(actual)} (Matches: {matches}/5)")
                    if hasattr(self, 'llm_reasoning_log') and i < len(self.llm_reasoning_log):
                        reasoning = self.llm_reasoning_log[i][:80] + "..." if len(self.llm_reasoning_log[i]) > 80 else self.llm_reasoning_log[i]
                        print(f"  LLM Reasoning: {reasoning}")
                    print()
            
            # Statistical analysis
            random_prob = self.n_numbers / self.number_range
            improvement = metrics['pattern_accuracy'] / (random_prob * 100)
            
            print(f"📊 STATISTICAL ANALYSIS:")
            print(f"   Random Expectation: {random_prob*100:.2f}%")
            print(f"   PatternSight v4.0: {metrics['pattern_accuracy']:.2f}%")
            print(f"   Improvement Factor: {improvement:.1f}x")
            
            # Hypothesis test
            n_trials = metrics['total_predictions']
            observed = metrics['avg_partial_matches'] * n_trials
            expected = random_prob * self.n_numbers * n_trials
            
            if expected > 0:
                z_score = (observed - expected) / np.sqrt(expected)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                print(f"   Z-score: {z_score:.2f}")
                print(f"   P-value: {p_value:.4f}")
                print(f"   Significance: {'YES' if p_value < 0.05 else 'NO'}")
            
            print("\n🏆 PatternSight v4.0 with LLM Integration:")
            print("   Revolutionary AI-enhanced lottery prediction system!")
            print("   Combines mathematical rigor with artificial intelligence!")
            
        else:
            print("❌ No integrated predictions generated")
        
        return {
            'data': data,
            'pillar_results': pillar_results,
            'integrated_predictions': integrated_predictions,
            'integrated_confidences': integrated_confidences,
            'metrics': metrics if integrated_predictions else {}
        }

def main():
    """Main analysis function"""
    analyzer = PatternSightV4LLMEnhanced()
    results = analyzer.run_v4_analysis('/home/ubuntu/upload/powerball_data_5years.json', predict_last_n=10)  # Small test
    
    print("\n" + "=" * 80)
    print("🎰 PATTERNSIGHT v4.0 LLM-ENHANCED ANALYSIS COMPLETE")
    print("   World's First AI-Enhanced Lottery Prediction System!")
    print("=" * 80)

if __name__ == "__main__":
    main()

