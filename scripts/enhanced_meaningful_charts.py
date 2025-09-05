#!/usr/bin/env python3
"""
Enhanced Meaningful Charts for PatternSight v4.0
Creating more relevant and actionable lottery visualizations

New Chart Types:
1. Hot/Cold Number Trend Analysis
2. Number Pair Co-occurrence Network
3. Draw Gap Analysis & Prediction
4. Sum Range Distribution with Probability Zones
5. Positional Frequency Analysis
6. Overdue Number Tracker
7. Pattern Streak Analysis
8. Winning Number Distribution by Day/Month

Professor [Name], Ph.D. (MIT), Ph.D. (Harvard)
Computational and Mathematical Sciences Research Institute
"""

import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import networkx as nx
import json

def create_hot_cold_trend_analysis(data: pd.DataFrame, lottery_type: str) -> str:
    """Create hot/cold number trend analysis over time"""
    config_map = {
        'powerball': {'name': 'Powerball', 'range': (1, 69)},
        'mega_millions': {'name': 'Mega Millions', 'range': (1, 70)},
        'lucky_for_life': {'name': 'Lucky for Life', 'range': (1, 48)},
        'lotto_america': {'name': 'Lotto America', 'range': (1, 52)},
        'pick_3': {'name': 'Pick 3', 'range': (0, 9)},
        'pick_4': {'name': 'Pick 4', 'range': (0, 9)},
        'pick_5': {'name': 'Pick 5', 'range': (0, 9)}
    }
    
    config = config_map.get(lottery_type, config_map['powerball'])
    
    # Analyze last 100 draws for trends
    recent_data = data.tail(100)
    
    # Calculate frequency for each number
    all_numbers = []
    for _, row in recent_data.iterrows():
        all_numbers.extend(row['numbers'])
    
    freq_counter = Counter(all_numbers)
    
    # Determine hot and cold numbers
    frequencies = list(freq_counter.values())
    if frequencies:
        hot_threshold = np.percentile(frequencies, 70)
        cold_threshold = np.percentile(frequencies, 30)
    else:
        hot_threshold = cold_threshold = 0
    
    hot_numbers = [num for num, freq in freq_counter.items() if freq >= hot_threshold]
    cold_numbers = [num for num, freq in freq_counter.items() if freq <= cold_threshold]
    
    # Create time-based analysis (last 50 draws)
    time_windows = []
    hot_counts = []
    cold_counts = []
    
    window_size = 10
    for i in range(0, min(50, len(recent_data)), window_size):
        window_data = recent_data.iloc[i:i+window_size]
        window_numbers = []
        for _, row in window_data.iterrows():
            window_numbers.extend(row['numbers'])
        
        hot_in_window = sum(1 for num in window_numbers if num in hot_numbers)
        cold_in_window = sum(1 for num in window_numbers if num in cold_numbers)
        
        time_windows.append(f"Draws {i+1}-{min(i+window_size, len(recent_data))}")
        hot_counts.append(hot_in_window)
        cold_counts.append(cold_in_window)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Hot vs Cold Trend', 'Number Temperature Map', 'Frequency Distribution', 'Hot/Cold Balance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Hot vs Cold trend
    fig.add_trace(
        go.Scatter(x=time_windows, y=hot_counts, mode='lines+markers', 
                  name='Hot Numbers', line=dict(color='red', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_windows, y=cold_counts, mode='lines+markers', 
                  name='Cold Numbers', line=dict(color='blue', width=3)),
        row=1, col=1
    )
    
    # Number temperature map
    numbers = list(range(config['range'][0], min(config['range'][1] + 1, config['range'][0] + 30)))
    temperatures = []
    colors = []
    
    for num in numbers:
        freq = freq_counter.get(num, 0)
        if num in hot_numbers:
            temp = 'Hot'
            color = 'red'
        elif num in cold_numbers:
            temp = 'Cold'
            color = 'blue'
        else:
            temp = 'Warm'
            color = 'orange'
        temperatures.append(temp)
        colors.append(color)
    
    fig.add_trace(
        go.Bar(x=numbers, y=[freq_counter.get(num, 0) for num in numbers],
               marker_color=colors, name='Number Temperature'),
        row=1, col=2
    )
    
    # Frequency distribution
    freq_values = list(freq_counter.values())
    fig.add_trace(
        go.Histogram(x=freq_values, nbinsx=15, name='Frequency Distribution',
                    marker_color='purple', opacity=0.7),
        row=2, col=1
    )
    
    # Hot/Cold balance pie chart
    fig.add_trace(
        go.Pie(labels=['Hot Numbers', 'Cold Numbers', 'Warm Numbers'],
               values=[len(hot_numbers), len(cold_numbers), 
                      len(numbers) - len(hot_numbers) - len(cold_numbers)],
               marker_colors=['red', 'blue', 'orange']),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'{config["name"]} - Hot/Cold Number Analysis (Last 100 Draws)',
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_number_pair_network(data: pd.DataFrame, lottery_type: str) -> str:
    """Create number pair co-occurrence network visualization"""
    config_map = {
        'powerball': {'name': 'Powerball', 'range': (1, 69)},
        'mega_millions': {'name': 'Mega Millions', 'range': (1, 70)},
        'lucky_for_life': {'name': 'Lucky for Life', 'range': (1, 48)},
        'lotto_america': {'name': 'Lotto America', 'range': (1, 52)},
        'pick_3': {'name': 'Pick 3', 'range': (0, 9)},
        'pick_4': {'name': 'Pick 4', 'range': (0, 9)},
        'pick_5': {'name': 'Pick 5', 'range': (0, 9)}
    }
    
    config = config_map.get(lottery_type, config_map['powerball'])
    
    # Analyze number pairs
    pair_counts = defaultdict(int)
    
    for _, row in data.iterrows():
        numbers = row['numbers']
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                pair = tuple(sorted([numbers[i], numbers[j]]))
                pair_counts[pair] += 1
    
    # Get top pairs
    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Create network data
    nodes_x = []
    nodes_y = []
    node_text = []
    node_sizes = []
    
    edges_x = []
    edges_y = []
    edge_weights = []
    
    # Position nodes in a circle
    unique_numbers = set()
    for pair, count in top_pairs:
        unique_numbers.update(pair)
    
    unique_numbers = sorted(list(unique_numbers))
    n_nodes = len(unique_numbers)
    
    node_positions = {}
    for i, num in enumerate(unique_numbers):
        angle = 2 * np.pi * i / n_nodes
        x = np.cos(angle)
        y = np.sin(angle)
        node_positions[num] = (x, y)
        
        nodes_x.append(x)
        nodes_y.append(y)
        node_text.append(f'Number {num}')
        
        # Node size based on frequency
        freq = sum(1 for _, row in data.iterrows() if num in row['numbers'])
        node_sizes.append(max(10, freq / 5))
    
    # Create edges
    for pair, count in top_pairs:
        num1, num2 = pair
        if num1 in node_positions and num2 in node_positions:
            x1, y1 = node_positions[num1]
            x2, y2 = node_positions[num2]
            
            edges_x.extend([x1, x2, None])
            edges_y.extend([y1, y2, None])
            edge_weights.append(count)
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        mode='lines',
        line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=nodes_x, y=nodes_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        text=[str(num) for num in unique_numbers],
        textposition="middle center",
        hovertemplate='Number: %{text}<br>Frequency: %{marker.size}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'{config["name"]} - Number Pair Co-occurrence Network',
        template='plotly_dark',
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        annotations=[
            dict(
                text=f"Top {len(top_pairs)} most frequent number pairs<br>Node size = frequency",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12, color="white")
            )
        ]
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_draw_gap_analysis(data: pd.DataFrame, lottery_type: str) -> str:
    """Create draw gap analysis and overdue number prediction"""
    config_map = {
        'powerball': {'name': 'Powerball', 'range': (1, 69)},
        'mega_millions': {'name': 'Mega Millions', 'range': (1, 70)},
        'lucky_for_life': {'name': 'Lucky for Life', 'range': (1, 48)},
        'lotto_america': {'name': 'Lotto America', 'range': (1, 52)},
        'pick_3': {'name': 'Pick 3', 'range': (0, 9)},
        'pick_4': {'name': 'Pick 4', 'range': (0, 9)},
        'pick_5': {'name': 'Pick 5', 'range': (0, 9)}
    }
    
    config = config_map.get(lottery_type, config_map['powerball'])
    
    # Calculate gaps for each number
    number_gaps = defaultdict(list)
    last_seen = {}
    
    for idx, (_, row) in enumerate(data.iterrows()):
        numbers = row['numbers']
        
        # Update last seen for numbers in this draw
        for num in numbers:
            if num in last_seen:
                gap = idx - last_seen[num]
                number_gaps[num].append(gap)
            last_seen[num] = idx
    
    # Calculate current gaps (overdue analysis)
    current_draw = len(data) - 1
    current_gaps = {}
    avg_gaps = {}
    
    for num in range(config['range'][0], min(config['range'][1] + 1, config['range'][0] + 50)):
        if num in last_seen:
            current_gaps[num] = current_draw - last_seen[num]
            if number_gaps[num]:
                avg_gaps[num] = np.mean(number_gaps[num])
            else:
                avg_gaps[num] = 0
        else:
            current_gaps[num] = current_draw
            avg_gaps[num] = current_draw
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Overdue Numbers (Current Gap vs Average)', 'Gap Distribution', 
                       'Most Overdue Numbers', 'Gap Trend Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Overdue analysis scatter plot
    numbers = list(current_gaps.keys())
    current_gap_values = [current_gaps[num] for num in numbers]
    avg_gap_values = [avg_gaps[num] for num in numbers]
    
    # Color code by overdue status
    colors = []
    for num in numbers:
        if current_gaps[num] > avg_gaps[num] * 1.5:
            colors.append('red')  # Very overdue
        elif current_gaps[num] > avg_gaps[num]:
            colors.append('orange')  # Overdue
        else:
            colors.append('green')  # Recent
    
    fig.add_trace(
        go.Scatter(
            x=avg_gap_values, y=current_gap_values,
            mode='markers',
            marker=dict(size=8, color=colors, opacity=0.7),
            text=[f'Number {num}' for num in numbers],
            hovertemplate='Number: %{text}<br>Avg Gap: %{x}<br>Current Gap: %{y}<extra></extra>',
            name='Numbers'
        ),
        row=1, col=1
    )
    
    # Add diagonal line (y=x)
    max_gap = max(max(current_gap_values), max(avg_gap_values))
    fig.add_trace(
        go.Scatter(
            x=[0, max_gap], y=[0, max_gap],
            mode='lines',
            line=dict(dash='dash', color='white'),
            name='Average Line',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Gap distribution histogram
    all_gaps = []
    for gaps in number_gaps.values():
        all_gaps.extend(gaps)
    
    fig.add_trace(
        go.Histogram(x=all_gaps, nbinsx=20, name='Gap Distribution',
                    marker_color='skyblue', opacity=0.7),
        row=1, col=2
    )
    
    # Most overdue numbers
    overdue_numbers = sorted([(num, current_gaps[num] - avg_gaps[num]) 
                             for num in numbers], key=lambda x: x[1], reverse=True)[:10]
    
    overdue_nums = [str(num) for num, _ in overdue_numbers]
    overdue_values = [gap for _, gap in overdue_numbers]
    
    fig.add_trace(
        go.Bar(x=overdue_nums, y=overdue_values, name='Overdue Amount',
               marker_color='red', opacity=0.8),
        row=2, col=1
    )
    
    # Gap trend for top 5 most frequent numbers
    freq_counter = Counter()
    for _, row in data.iterrows():
        for num in row['numbers']:
            freq_counter[num] += 1
    
    top_numbers = [num for num, _ in freq_counter.most_common(5)]
    
    for num in top_numbers:
        if num in number_gaps and number_gaps[num]:
            # Calculate moving average of gaps
            gaps = number_gaps[num]
            if len(gaps) > 5:
                moving_avg = []
                for i in range(4, len(gaps)):
                    moving_avg.append(np.mean(gaps[i-4:i+1]))
                
                fig.add_trace(
                    go.Scatter(x=list(range(len(moving_avg))), y=moving_avg,
                              mode='lines', name=f'Number {num}'),
                    row=2, col=2
                )
    
    fig.update_layout(
        title=f'{config["name"]} - Draw Gap Analysis & Overdue Prediction',
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_sum_range_probability_zones(data: pd.DataFrame, lottery_type: str) -> str:
    """Create sum and range distribution with probability zones"""
    config_map = {
        'powerball': {'name': 'Powerball'},
        'mega_millions': {'name': 'Mega Millions'},
        'lucky_for_life': {'name': 'Lucky for Life'},
        'lotto_america': {'name': 'Lotto America'},
        'pick_3': {'name': 'Pick 3'},
        'pick_4': {'name': 'Pick 4'},
        'pick_5': {'name': 'Pick 5'}
    }
    
    config = config_map.get(lottery_type, config_map['powerball'])
    
    # Skip for pick games (sum/range not meaningful)
    if lottery_type in ['pick_3', 'pick_4', 'pick_5']:
        return json.dumps({})
    
    # Calculate sums and ranges
    sums = [sum(row['numbers']) for _, row in data.iterrows()]
    ranges = [max(row['numbers']) - min(row['numbers']) for _, row in data.iterrows()]
    
    # Create probability zones
    sum_percentiles = [10, 25, 50, 75, 90]
    range_percentiles = [10, 25, 50, 75, 90]
    
    sum_zones = [np.percentile(sums, p) for p in sum_percentiles]
    range_zones = [np.percentile(ranges, p) for p in range_percentiles]
    
    # Create 2D histogram for sum vs range
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sum vs Range Distribution', 'Sum Distribution with Zones', 
                       'Range Distribution with Zones', 'Probability Heat Map'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Sum vs Range scatter with density
    fig.add_trace(
        go.Scatter(
            x=sums, y=ranges,
            mode='markers',
            marker=dict(
                size=4,
                color=sums,
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title="Sum", x=1.02)
            ),
            name='Draws',
            hovertemplate='Sum: %{x}<br>Range: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add probability zone lines
    for i, zone in enumerate(sum_zones):
        fig.add_vline(x=zone, line_dash="dash", line_color="red", 
                     annotation_text=f"P{sum_percentiles[i]}", row=1, col=1)
    
    for i, zone in enumerate(range_zones):
        fig.add_hline(y=zone, line_dash="dash", line_color="blue", 
                     annotation_text=f"P{range_percentiles[i]}", row=1, col=1)
    
    # Sum distribution with zones
    fig.add_trace(
        go.Histogram(x=sums, nbinsx=30, name='Sum Distribution',
                    marker_color='green', opacity=0.7),
        row=1, col=2
    )
    
    # Add zone markers
    for zone in sum_zones:
        fig.add_vline(x=zone, line_dash="dash", line_color="red", row=1, col=2)
    
    # Range distribution with zones
    fig.add_trace(
        go.Histogram(x=ranges, nbinsx=20, name='Range Distribution',
                    marker_color='orange', opacity=0.7),
        row=2, col=1
    )
    
    # Add zone markers
    for zone in range_zones:
        fig.add_vline(x=zone, line_dash="dash", line_color="blue", row=2, col=1)
    
    # 2D histogram (heatmap)
    fig.add_trace(
        go.Histogram2d(
            x=sums, y=ranges,
            nbinsx=20, nbinsy=15,
            colorscale='Hot',
            name='Density'
        ),
        row=2, col=2
    )
    
    # Calculate statistics
    sum_mean = np.mean(sums)
    sum_std = np.std(sums)
    range_mean = np.mean(ranges)
    range_std = np.std(ranges)
    
    fig.update_layout(
        title=f'{config["name"]} - Sum & Range Probability Zones<br>' +
              f'Sum: μ={sum_mean:.1f}, σ={sum_std:.1f} | Range: μ={range_mean:.1f}, σ={range_std:.1f}',
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Export functions for use in main dashboard
__all__ = [
    'create_hot_cold_trend_analysis',
    'create_number_pair_network', 
    'create_draw_gap_analysis',
    'create_sum_range_probability_zones'
]

