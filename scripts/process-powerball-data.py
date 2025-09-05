#!/usr/bin/env python3
"""
Process and Analyze 5 Years of Powerball Historical Data
Prepare data for AI model fine-tuning
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
import re

# Load the scraped data
with open('powerball_data_5years.json', 'r') as f:
    raw_data = json.load(f)

print(f"📊 POWERBALL DATA ANALYSIS (5 Years)")
print("="*80)
print(f"Total records: {len(raw_data)}")

# Process the data
processed_data = []
for entry in raw_data:
    # Parse the winning numbers string
    numbers_str = entry['winning_numbers']
    numbers = [int(n) for n in numbers_str.split()]
    
    # Separate main numbers and powerball
    main_numbers = sorted(numbers[:5])
    powerball = numbers[5]
    
    # Parse date
    date = datetime.strptime(entry['draw_date'][:10], '%Y-%m-%d')
    
    processed_data.append({
        'date': date,
        'year': date.year,
        'month': date.month,
        'day_of_week': date.weekday(),
        'main_numbers': main_numbers,
        'powerball': powerball,
        'multiplier': int(entry.get('multiplier', 1)),
        'sum_of_numbers': sum(main_numbers),
        'avg_of_numbers': np.mean(main_numbers),
        'std_of_numbers': np.std(main_numbers),
        'range_of_numbers': max(main_numbers) - min(main_numbers),
        'consecutive_count': sum(1 for i in range(len(main_numbers)-1) if main_numbers[i+1] - main_numbers[i] == 1),
        'odd_count': sum(1 for n in main_numbers if n % 2 == 1),
        'even_count': sum(1 for n in main_numbers if n % 2 == 0),
        'low_count': sum(1 for n in main_numbers if n <= 34),
        'high_count': sum(1 for n in main_numbers if n > 34),
        'decade_distribution': [
            sum(1 for n in main_numbers if 1 <= n <= 9),
            sum(1 for n in main_numbers if 10 <= n <= 19),
            sum(1 for n in main_numbers if 20 <= n <= 29),
            sum(1 for n in main_numbers if 30 <= n <= 39),
            sum(1 for n in main_numbers if 40 <= n <= 49),
            sum(1 for n in main_numbers if 50 <= n <= 59),
            sum(1 for n in main_numbers if 60 <= n <= 69)
        ]
    })

# Create DataFrame
df = pd.DataFrame(processed_data)
df = df.sort_values('date')

print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Draws analyzed: {len(df)}")

# Statistical Analysis
print("\n📈 STATISTICAL ANALYSIS")
print("="*80)

# Frequency analysis for main numbers
all_main_numbers = []
for nums in df['main_numbers']:
    all_main_numbers.extend(nums)

main_freq = Counter(all_main_numbers)
top_20_main = main_freq.most_common(20)
bottom_20_main = main_freq.most_common()[-20:]

print("\n🔥 Top 20 HOT Numbers (Most Frequent):")
for i, (num, count) in enumerate(top_20_main, 1):
    pct = (count / len(df) * 100 / 5)
    print(f"  {i:2d}. Number {num:02d}: {count:3d} times ({pct:5.2f}%)")

print("\n❄️ Bottom 20 COLD Numbers (Least Frequent):")
for i, (num, count) in enumerate(bottom_20_main, 1):
    pct = (count / len(df) * 100 / 5)
    print(f"  {i:2d}. Number {num:02d}: {count:3d} times ({pct:5.2f}%)")

# Powerball frequency
powerball_freq = Counter(df['powerball'])
top_10_powerball = powerball_freq.most_common(10)

print("\n⚡ Top 10 Powerball Numbers:")
for i, (num, count) in enumerate(top_10_powerball, 1):
    pct = (count / len(df) * 100)
    print(f"  {i:2d}. Powerball {num:02d}: {count:3d} times ({pct:5.2f}%)")

# Pattern Analysis
print("\n🔍 PATTERN ANALYSIS")
print("="*80)

# Consecutive numbers analysis
consecutive_stats = df['consecutive_count'].value_counts().sort_index()
print("\nConsecutive Numbers Pattern:")
for count, freq in consecutive_stats.items():
    pct = (freq / len(df) * 100)
    print(f"  {count} consecutive: {freq:3d} times ({pct:5.2f}%)")

# Odd/Even distribution
print("\nOdd/Even Distribution:")
odd_even_patterns = df.groupby(['odd_count', 'even_count']).size().reset_index(name='frequency')
odd_even_patterns = odd_even_patterns.sort_values('frequency', ascending=False)
for _, row in odd_even_patterns.head(10).iterrows():
    pct = (row['frequency'] / len(df) * 100)
    print(f"  {int(row['odd_count'])} odd, {int(row['even_count'])} even: {row['frequency']:3d} times ({pct:5.2f}%)")

# Sum ranges
print("\nSum of Numbers Distribution:")
sum_stats = df['sum_of_numbers'].describe()
print(f"  Mean: {sum_stats['mean']:.1f}")
print(f"  Std:  {sum_stats['std']:.1f}")
print(f"  Min:  {sum_stats['min']:.0f}")
print(f"  Max:  {sum_stats['max']:.0f}")

# Time-based patterns
print("\n📅 TIME-BASED PATTERNS")
print("="*80)

# Day of week analysis
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_freq = df['day_of_week'].value_counts().sort_index()

print("\nDraws by Day of Week:")
for day_num, count in day_freq.items():
    if day_num < len(day_names):
        print(f"  {day_names[day_num]:10s}: {count:3d} draws")

# Monthly patterns
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_freq = df['month'].value_counts().sort_index()

print("\nDraws by Month:")
for month_num, count in month_freq.items():
    print(f"  {month_names[month_num-1]:3s}: {count:3d} draws")

# Positional Analysis
print("\n📍 POSITIONAL ANALYSIS")
print("="*80)

position_freq = defaultdict(Counter)
for nums in df['main_numbers']:
    for pos, num in enumerate(nums):
        position_freq[pos][num] += 1

print("\nMost Common Numbers by Position:")
for pos in range(5):
    top_5 = position_freq[pos].most_common(5)
    nums_str = ', '.join([f"{n:02d}({c})" for n, c in top_5])
    print(f"  Position {pos+1}: {nums_str}")

# Gap Analysis
print("\n🔢 GAP ANALYSIS")
print("="*80)

# Calculate gaps between appearances for each number
number_gaps = defaultdict(list)
number_last_seen = {}

for idx, row in df.iterrows():
    draw_idx = idx
    for num in row['main_numbers']:
        if num in number_last_seen:
            gap = draw_idx - number_last_seen[num]
            number_gaps[num].append(gap)
        number_last_seen[num] = draw_idx

# Average gaps
avg_gaps = {}
for num, gaps in number_gaps.items():
    if gaps:
        avg_gaps[num] = np.mean(gaps)

# Sort by average gap
sorted_gaps = sorted(avg_gaps.items(), key=lambda x: x[1])

print("\nNumbers with Shortest Average Gaps (appear frequently):")
for num, gap in sorted_gaps[:10]:
    print(f"  Number {num:02d}: avg gap of {gap:.1f} draws")

print("\nNumbers with Longest Average Gaps (appear rarely):")
for num, gap in sorted_gaps[-10:]:
    print(f"  Number {num:02d}: avg gap of {gap:.1f} draws")

# Recent Trends (Last 50 draws)
print("\n🔄 RECENT TRENDS (Last 50 Draws)")
print("="*80)

recent_df = df.tail(50)
recent_main = []
for nums in recent_df['main_numbers']:
    recent_main.extend(nums)

recent_freq = Counter(recent_main)
recent_top10 = recent_freq.most_common(10)

print("\nHottest Numbers in Last 50 Draws:")
for i, (num, count) in enumerate(recent_top10, 1):
    pct = (count / 50 * 100 / 5)
    print(f"  {i:2d}. Number {num:02d}: {count:2d} times ({pct:5.2f}%)")

# Prepare AI Training Data
print("\n🤖 PREPARING AI TRAINING DATA")
print("="*80)

# Create feature matrix for AI training
features = []
targets = []

for i in range(10, len(df) - 1):  # Use 10 draws as context
    # Features: last 10 draws
    feature_vector = []
    
    for j in range(10):
        idx = i - j - 1
        draw = df.iloc[idx]
        
        # Add various features
        feature_vector.extend(draw['main_numbers'])
        feature_vector.append(draw['powerball'])
        feature_vector.append(draw['sum_of_numbers'])
        feature_vector.append(draw['avg_of_numbers'])
        feature_vector.append(draw['odd_count'])
        feature_vector.append(draw['consecutive_count'])
        feature_vector.extend(draw['decade_distribution'])
    
    features.append(feature_vector)
    
    # Target: next draw
    next_draw = df.iloc[i]
    target = next_draw['main_numbers'] + [next_draw['powerball']]
    targets.append(target)

# Convert to numpy arrays
X = np.array(features)
y = np.array(targets)

print(f"Training samples: {len(X)}")
print(f"Feature dimensions: {X.shape[1]}")
print(f"Target dimensions: {y.shape[1]}")

# Save processed data
training_data = {
    'features': X.tolist(),
    'targets': y.tolist(),
    'feature_names': [
        'historical_numbers', 'historical_powerballs', 'sums', 'averages',
        'odd_counts', 'consecutive_counts', 'decade_distributions'
    ],
    'statistics': {
        'hot_numbers': dict(top_20_main),
        'cold_numbers': dict(bottom_20_main),
        'hot_powerballs': dict(top_10_powerball),
        'avg_sum': float(sum_stats['mean']),
        'std_sum': float(sum_stats['std']),
        'total_draws': len(df),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        }
    },
    'patterns': {
        'consecutive_frequency': consecutive_stats.to_dict(),
        'odd_even_patterns': odd_even_patterns.head(10).to_dict('records'),
        'position_favorites': {f'position_{p+1}': dict(position_freq[p].most_common(10)) for p in range(5)},
        'recent_trends': dict(recent_top10),
        'gap_analysis': {
            'short_gaps': dict(sorted_gaps[:10]),
            'long_gaps': dict(sorted_gaps[-10:])
        }
    }
}

# Save to JSON
with open('powerball_ai_training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)

print("\n✅ AI Training Data Saved to: powerball_ai_training_data.json")

# Generate summary report
summary = {
    'total_draws': len(df),
    'years_covered': df['year'].nunique(),
    'most_common_number': top_20_main[0][0],
    'least_common_number': bottom_20_main[0][0],
    'most_common_powerball': top_10_powerball[0][0],
    'avg_sum': float(sum_stats['mean']),
    'most_common_pattern': f"{odd_even_patterns.iloc[0]['odd_count']:.0f} odd, {odd_even_patterns.iloc[0]['even_count']:.0f} even",
    'consecutive_probability': float((df['consecutive_count'] > 0).mean() * 100)
}

print("\n📋 SUMMARY")
print("="*80)
for key, value in summary.items():
    print(f"  {key.replace('_', ' ').title():30s}: {value}")

print("\n✅ Data processing complete!")
print("Ready for AI model fine-tuning with 5 years of historical patterns.")