#!/usr/bin/env python3
"""
PatternSight v4.0 Enhanced Multi-Lottery Dashboard
The Ultimate Lottery Prediction Platform with Advanced Visualizations

Enhanced Features:
- 7 Lottery Systems: Powerball, Mega Millions, Lucky for Life, Pick 3, Pick 4, Pick 5, Lotto America
- Advanced Heatmaps and Correlation Analysis
- 3D Visualizations and Interactive Charts
- Real-time Pattern Recognition
- AI-Enhanced Predictions with Stunning UI

Professor [Name], Ph.D. (MIT), Ph.D. (Harvard)
Computational and Mathematical Sciences Research Institute
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.express as px
import plotly.utils
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from fixed_data_loader import load_real_lottery_data
from patternsight_unified_system import PatternSightV4Unified, PredictionResult, LotteryConfig
from enhanced_meaningful_charts import (
    create_hot_cold_trend_analysis,
    create_number_pair_network,
    create_draw_gap_analysis,
    create_sum_range_probability_zones
)
import os
import logging
from typing import Dict, List, Any
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'patternsight_v4_enhanced_dashboard_2025'

class EnhancedPatternSight(PatternSightV4Unified):
    """Enhanced PatternSight with additional lottery systems and visualizations"""
    
    def __init__(self):
        super().__init__()
        self.lottery_configs = self.initialize_enhanced_lottery_configs()
        logger.info(f"🚀 Enhanced PatternSight v4.0 with {len(self.lottery_configs)} lottery systems")
    
    def initialize_enhanced_lottery_configs(self) -> Dict[str, LotteryConfig]:
        """Initialize configurations for all requested lottery systems"""
        configs = {
            'powerball': LotteryConfig(
                name="Powerball",
                main_numbers=5,
                main_range=(1, 69),
                bonus_numbers=1,
                bonus_range=(1, 26),
                country="US"
            ),
            'mega_millions': LotteryConfig(
                name="Mega Millions",
                main_numbers=5,
                main_range=(1, 70),
                bonus_numbers=1,
                bonus_range=(1, 25),
                country="US"
            ),
            'lucky_for_life': LotteryConfig(
                name="Lucky for Life",
                main_numbers=5,
                main_range=(1, 48),
                bonus_numbers=1,
                bonus_range=(1, 18),
                country="US"
            ),
            'pick_3': LotteryConfig(
                name="Pick 3",
                main_numbers=3,
                main_range=(0, 9),
                bonus_numbers=0,
                bonus_range=(0, 0),
                country="US"
            ),
            'pick_4': LotteryConfig(
                name="Pick 4",
                main_numbers=4,
                main_range=(0, 9),
                bonus_numbers=0,
                bonus_range=(0, 0),
                country="US"
            ),
            'pick_5': LotteryConfig(
                name="Pick 5",
                main_numbers=5,
                main_range=(0, 9),
                bonus_numbers=0,
                bonus_range=(0, 0),
                country="US"
            ),
            'lotto_america': LotteryConfig(
                name="Lotto America",
                main_numbers=5,
                main_range=(1, 52),
                bonus_numbers=1,
                bonus_range=(1, 10),
                country="US"
            )
        }
        return configs

# Initialize enhanced system
enhanced_patternsight = EnhancedPatternSight()

# Global data storage
lottery_data = {}
prediction_history = {}

def load_lottery_data_from_file(file_path: str, lottery_type: str) -> pd.DataFrame:
    """Load lottery data from provided files"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        draws = []
        config = enhanced_patternsight.lottery_configs[lottery_type]
        
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
                        
                        if lottery_type in ['pick_3', 'pick_4', 'pick_5']:
                            # Handle Pick games (digits 0-9)
                            numbers = [int(x) for x in numbers_str.replace(' ', '')][:config.main_numbers]
                        else:
                            # Handle ball games
                            numbers = [int(x) for x in numbers_str.split()]
                            main_numbers = sorted(numbers[:config.main_numbers])
                            bonus_number = numbers[config.main_numbers] if len(numbers) > config.main_numbers and config.bonus_numbers > 0 else None
                        
                        draw_entry = {
                            'date': draw_date,
                            'numbers': numbers if lottery_type in ['pick_3', 'pick_4', 'pick_5'] else main_numbers,
                            'day_of_week': draw_date.strftime('%A'),
                            'month': draw_date.month,
                            'year': draw_date.year
                        }
                        
                        if lottery_type not in ['pick_3', 'pick_4', 'pick_5'] and config.bonus_numbers > 0:
                            draw_entry['bonus'] = bonus_number
                        
                        draws.append(draw_entry)
                        
                except json.JSONDecodeError:
                    continue
        
        draws.sort(key=lambda x: x['date'])
        df = pd.DataFrame(draws)
        
        logger.info(f"✅ Loaded {len(draws)} {lottery_type} draws")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load {lottery_type} data: {e}")
        return pd.DataFrame()

def create_number_heatmap(data: pd.DataFrame, lottery_type: str) -> str:
    """Create stunning number frequency heatmap"""
    config = enhanced_patternsight.lottery_configs[lottery_type]
    
    if lottery_type in ['pick_3', 'pick_4', 'pick_5']:
        # For Pick games, create position-based heatmap
        position_data = np.zeros((config.main_numbers, 10))
        
        for _, row in data.iterrows():
            for pos, digit in enumerate(row['numbers']):
                if pos < config.main_numbers:
                    position_data[pos][digit] += 1
        
        fig = go.Figure(data=go.Heatmap(
            z=position_data,
            x=list(range(10)),
            y=[f'Position {i+1}' for i in range(config.main_numbers)],
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Position: %{y}<br>Digit: %{x}<br>Frequency: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{config.name} - Position-Digit Frequency Heatmap',
            xaxis_title='Digits (0-9)',
            yaxis_title='Positions',
            template='plotly_dark',
            height=400
        )
    else:
        # For ball games, create number frequency grid
        all_numbers = [num for _, row in data.iterrows() for num in row['numbers']]
        freq_counter = Counter(all_numbers)
        
        # Create grid layout
        grid_size = int(np.ceil(np.sqrt(config.main_range[1])))
        heatmap_data = np.zeros((grid_size, grid_size))
        
        for num in range(config.main_range[0], config.main_range[1] + 1):
            row = (num - 1) // grid_size
            col = (num - 1) % grid_size
            if row < grid_size and col < grid_size:
                heatmap_data[row][col] = freq_counter.get(num, 0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            colorscale='Plasma',
            hoverongaps=False,
            hovertemplate='Frequency: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{config.name} - Number Frequency Heatmap',
            template='plotly_dark',
            height=500,
            showlegend=False
        )
        
        # Remove axis labels for cleaner look
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_correlation_matrix(data: pd.DataFrame, lottery_type: str) -> str:
    """Create number correlation matrix visualization"""
    config = enhanced_patternsight.lottery_configs[lottery_type]
    
    if len(data) < 50:  # Need sufficient data for correlation
        return json.dumps({})
    
    # Create co-occurrence matrix
    if lottery_type in ['pick_3', 'pick_4', 'pick_5']:
        # For Pick games, analyze digit co-occurrence
        cooccurrence = np.zeros((10, 10))
        
        for _, row in data.iterrows():
            numbers = row['numbers']
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    if i < len(numbers) and j < len(numbers):
                        digit1, digit2 = numbers[i], numbers[j]
                        cooccurrence[digit1][digit2] += 1
                        cooccurrence[digit2][digit1] += 1
        
        labels = [str(i) for i in range(10)]
    else:
        # For ball games, analyze number co-occurrence
        max_num = min(config.main_range[1], 50)  # Limit for visualization
        cooccurrence = np.zeros((max_num, max_num))
        
        for _, row in data.iterrows():
            numbers = row['numbers']
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    if numbers[i] <= max_num and numbers[j] <= max_num:
                        num1, num2 = numbers[i] - 1, numbers[j] - 1
                        cooccurrence[num1][num2] += 1
                        cooccurrence[num2][num1] += 1
        
        labels = [str(i+1) for i in range(max_num)]
    
    # Normalize correlation matrix
    correlation = np.corrcoef(cooccurrence)
    correlation = np.nan_to_num(correlation)
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmid=0,
        hoverongaps=False,
        hovertemplate='Number 1: %{y}<br>Number 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{config.name} - Number Correlation Matrix',
        template='plotly_dark',
        height=500,
        xaxis_title='Numbers',
        yaxis_title='Numbers'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_3d_pattern_analysis(data: pd.DataFrame, lottery_type: str) -> str:
    """Create stunning 3D pattern visualization"""
    config = enhanced_patternsight.lottery_configs[lottery_type]
    
    if len(data) < 100:
        return json.dumps({})
    
    # Analyze patterns over time
    recent_data = data.tail(200)  # Last 200 draws
    
    if lottery_type in ['pick_3', 'pick_4', 'pick_5']:
        # For Pick games, create 3D digit frequency over time
        time_windows = 20
        window_size = len(recent_data) // time_windows
        
        x_time = []
        y_digits = []
        z_frequency = []
        
        for window in range(time_windows):
            start_idx = window * window_size
            end_idx = min((window + 1) * window_size, len(recent_data))
            window_data = recent_data.iloc[start_idx:end_idx]
            
            digit_counts = Counter()
            for _, row in window_data.iterrows():
                for digit in row['numbers']:
                    digit_counts[digit] += 1
            
            for digit in range(10):
                x_time.append(window)
                y_digits.append(digit)
                z_frequency.append(digit_counts.get(digit, 0))
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_time,
            y=y_digits,
            z=z_frequency,
            mode='markers',
            marker=dict(
                size=8,
                color=z_frequency,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Frequency")
            ),
            hovertemplate='Time Window: %{x}<br>Digit: %{y}<br>Frequency: %{z}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'{config.name} - 3D Digit Frequency Evolution',
            scene=dict(
                xaxis_title='Time Windows',
                yaxis_title='Digits',
                zaxis_title='Frequency'
            ),
            template='plotly_dark',
            height=600
        )
    else:
        # For ball games, create 3D sum-range-frequency analysis
        sums = [sum(row['numbers']) for _, row in recent_data.iterrows()]
        ranges = [max(row['numbers']) - min(row['numbers']) for _, row in recent_data.iterrows()]
        dates = list(range(len(recent_data)))
        
        fig = go.Figure(data=[go.Scatter3d(
            x=dates,
            y=sums,
            z=ranges,
            mode='markers+lines',
            marker=dict(
                size=6,
                color=sums,
                colorscale='Plasma',
                opacity=0.8,
                colorbar=dict(title="Sum")
            ),
            line=dict(
                color='rgba(255,255,255,0.3)',
                width=2
            ),
            hovertemplate='Draw: %{x}<br>Sum: %{y}<br>Range: %{z}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'{config.name} - 3D Sum-Range Pattern Evolution',
            scene=dict(
                xaxis_title='Draw Number',
                yaxis_title='Sum',
                zaxis_title='Range'
            ),
            template='plotly_dark',
            height=600
        )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_advanced_frequency_chart(data: pd.DataFrame, lottery_type: str) -> str:
    """Create advanced frequency analysis with multiple views"""
    config = enhanced_patternsight.lottery_configs[lottery_type]
    
    # Create subplot with multiple frequency analyses
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Overall Frequency', 'Recent Trends', 'Hot vs Cold', 'Frequency Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if lottery_type in ['pick_3', 'pick_4', 'pick_5']:
        # Overall frequency
        all_digits = [digit for _, row in data.iterrows() for digit in row['numbers']]
        digit_counts = Counter(all_digits)
        
        fig.add_trace(
            go.Bar(x=list(range(10)), y=[digit_counts.get(i, 0) for i in range(10)],
                   name='Overall Frequency', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Recent trends (last 100 draws)
        recent_data = data.tail(100)
        recent_digits = [digit for _, row in recent_data.iterrows() for digit in row['numbers']]
        recent_counts = Counter(recent_digits)
        
        fig.add_trace(
            go.Scatter(x=list(range(10)), y=[recent_counts.get(i, 0) for i in range(10)],
                      mode='lines+markers', name='Recent Trend', line_color='orange'),
            row=1, col=2
        )
        
        # Hot vs Cold analysis
        hot_threshold = np.percentile(list(digit_counts.values()), 70)
        hot_digits = [digit for digit, count in digit_counts.items() if count >= hot_threshold]
        cold_digits = [digit for digit, count in digit_counts.items() if count < hot_threshold]
        
        fig.add_trace(
            go.Bar(x=['Hot Digits', 'Cold Digits'], y=[len(hot_digits), len(cold_digits)],
                   marker_color=['red', 'blue'], name='Hot vs Cold'),
            row=2, col=1
        )
        
    else:
        # Similar analysis for ball games
        all_numbers = [num for _, row in data.iterrows() for num in row['numbers']]
        freq_counter = Counter(all_numbers)
        
        numbers = list(range(config.main_range[0], min(config.main_range[1] + 1, config.main_range[0] + 30)))
        frequencies = [freq_counter.get(num, 0) for num in numbers]
        
        fig.add_trace(
            go.Bar(x=numbers, y=frequencies, name='Overall Frequency', marker_color='lightgreen'),
            row=1, col=1
        )
        
        # Recent trends
        recent_data = data.tail(100)
        recent_numbers = [num for _, row in recent_data.iterrows() for num in row['numbers']]
        recent_counts = Counter(recent_numbers)
        recent_freq = [recent_counts.get(num, 0) for num in numbers]
        
        fig.add_trace(
            go.Scatter(x=numbers, y=recent_freq, mode='lines+markers', 
                      name='Recent Trend', line_color='purple'),
            row=1, col=2
        )
    
    # Frequency distribution histogram
    freq_values = list(freq_counter.values()) if 'freq_counter' in locals() else list(digit_counts.values())
    fig.add_trace(
        go.Histogram(x=freq_values, nbinsx=20, name='Frequency Distribution', 
                    marker_color='gold', opacity=0.7),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f'{config.name} - Advanced Frequency Analysis',
        template='plotly_dark',
        height=700,
        showlegend=True
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_enhanced_dashboard_template():
    """Create enhanced HTML template with advanced visualizations"""
    os.makedirs('templates', exist_ok=True)
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PatternSight v4.0 Enhanced - Multi-Lottery Prediction Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            font-family: 'Inter', sans-serif;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
        }
        .gradient-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .prediction-ball {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 18px;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            animation: glow 2s ease-in-out infinite alternate;
        }
        .bonus-ball {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
        }
        @keyframes glow {
            from { box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3); }
            to { box-shadow: 0 8px 32px rgba(102, 126, 234, 0.6); }
        }
        .loading {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        .pillar-contribution {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
        }
        .pillar-contribution:hover {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
            transform: translateX(5px);
        }
        .chart-container {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
        }
        .wow-effect {
            animation: fadeInUp 0.8s ease-out;
        }
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .lottery-tab {
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .lottery-tab:hover {
            background: rgba(102, 126, 234, 0.2);
            transform: scale(1.05);
        }
        .lottery-tab.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
    </style>
</head>
<body class="min-h-screen text-white">
    <!-- Header -->
    <header class="glass-card m-6 p-6 wow-effect">
        <div class="flex items-center justify-between">
            <div>
                <h1 class="text-5xl font-bold gradient-text">PatternSight v4.0 Enhanced</h1>
                <p class="text-gray-300 mt-2">🚀 The Ultimate Multi-Lottery Prediction Platform with Advanced AI Analytics</p>
                <div class="flex items-center mt-3 space-x-4">
                    <span class="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm">
                        <i class="fas fa-brain mr-1"></i>AI-Powered
                    </span>
                    <span class="bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full text-sm">
                        <i class="fas fa-chart-line mr-1"></i>Real-time Analytics
                    </span>
                    <span class="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm">
                        <i class="fas fa-eye mr-1"></i>Transparent AI
                    </span>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <div class="text-right">
                    <div class="text-sm text-gray-400">System Status</div>
                    <div class="text-green-400 font-semibold" id="system-status">🟢 Active</div>
                    <div class="text-xs text-gray-500" id="active-lotteries">7 Lottery Systems</div>
                </div>
                <button onclick="refreshSystemStatus()" class="glass-card p-3 hover:bg-white/10 transition-all">
                    <i class="fas fa-sync-alt"></i>
                </button>
            </div>
        </div>
    </header>

    <!-- Lottery System Tabs -->
    <div class="mx-6 mb-6">
        <div class="glass-card p-4">
            <h3 class="text-lg font-bold mb-4">🎰 Select Lottery System</h3>
            <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3" id="lottery-tabs">
                <div class="lottery-tab glass-card p-3 text-center active" data-lottery="powerball">
                    <i class="fas fa-circle text-red-400 mb-2"></i>
                    <div class="font-semibold">Powerball</div>
                    <div class="text-xs text-gray-400">5/69 + 1/26</div>
                </div>
                <div class="lottery-tab glass-card p-3 text-center" data-lottery="mega_millions">
                    <i class="fas fa-circle text-yellow-400 mb-2"></i>
                    <div class="font-semibold">Mega Millions</div>
                    <div class="text-xs text-gray-400">5/70 + 1/25</div>
                </div>
                <div class="lottery-tab glass-card p-3 text-center" data-lottery="lucky_for_life">
                    <i class="fas fa-circle text-green-400 mb-2"></i>
                    <div class="font-semibold">Lucky for Life</div>
                    <div class="text-xs text-gray-400">5/48 + 1/18</div>
                </div>
                <div class="lottery-tab glass-card p-3 text-center" data-lottery="pick_3">
                    <i class="fas fa-circle text-blue-400 mb-2"></i>
                    <div class="font-semibold">Pick 3</div>
                    <div class="text-xs text-gray-400">3 Digits</div>
                </div>
                <div class="lottery-tab glass-card p-3 text-center" data-lottery="pick_4">
                    <i class="fas fa-circle text-purple-400 mb-2"></i>
                    <div class="font-semibold">Pick 4</div>
                    <div class="text-xs text-gray-400">4 Digits</div>
                </div>
                <div class="lottery-tab glass-card p-3 text-center" data-lottery="pick_5">
                    <i class="fas fa-circle text-pink-400 mb-2"></i>
                    <div class="font-semibold">Pick 5</div>
                    <div class="text-xs text-gray-400">5 Digits</div>
                </div>
                <div class="lottery-tab glass-card p-3 text-center" data-lottery="lotto_america">
                    <i class="fas fa-circle text-orange-400 mb-2"></i>
                    <div class="font-semibold">Lotto America</div>
                    <div class="text-xs text-gray-400">5/52 + 1/10</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Dashboard -->
    <div class="mx-6 grid grid-cols-1 lg:grid-cols-4 gap-6">
        <!-- Control Panel -->
        <div class="lg:col-span-1">
            <div class="glass-card p-6 mb-6 wow-effect">
                <h2 class="text-2xl font-bold mb-4">🎯 AI Prediction Engine</h2>
                
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">Number of Predictions</label>
                    <input type="number" id="n-predictions" min="1" max="5" value="1" 
                           class="w-full bg-white/10 border border-white/20 rounded-lg p-3 text-white">
                </div>
                
                <button onclick="generatePredictions()" 
                        class="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 
                               text-white font-bold py-4 px-6 rounded-lg transition-all transform hover:scale-105 mb-4">
                    <i class="fas fa-magic mr-2"></i>Generate AI Predictions
                </button>
                
                <div class="text-center text-sm text-gray-400">
                    <i class="fas fa-robot mr-1"></i>
                    Powered by 10-Pillar AI Architecture
                </div>
            </div>

            <!-- System Information -->
            <div class="glass-card p-6 wow-effect">
                <h3 class="text-xl font-bold mb-4">🏗️ AI Pillars Status</h3>
                <div id="pillar-info" class="space-y-2">
                    <div class="pillar-contribution p-3 rounded-lg">
                        <div class="font-semibold">CDM Bayesian</div>
                        <div class="text-sm text-gray-400">Weight: 20% | Active</div>
                    </div>
                    <div class="pillar-contribution p-3 rounded-lg">
                        <div class="font-semibold">Order Statistics</div>
                        <div class="text-sm text-gray-400">Weight: 16% | Best Performer</div>
                    </div>
                    <div class="pillar-contribution p-3 rounded-lg">
                        <div class="font-semibold">Markov Chain</div>
                        <div class="text-sm text-gray-400">Weight: 14% | Active</div>
                    </div>
                    <div class="pillar-contribution p-3 rounded-lg">
                        <div class="font-semibold">LLM Reasoning</div>
                        <div class="text-sm text-gray-400">Weight: 18% + AI Bonus</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results and Analytics Panel -->
        <div class="lg:col-span-3">
            <!-- Predictions Display -->
            <div class="glass-card p-6 mb-6 wow-effect">
                <h2 class="text-2xl font-bold mb-4">🔮 AI-Generated Predictions</h2>
                <div id="predictions-container" class="text-center text-gray-400 py-8">
                    <i class="fas fa-crystal-ball text-6xl mb-4 text-purple-400"></i>
                    <p class="text-lg">Select a lottery system and generate AI-powered predictions</p>
                    <p class="text-sm mt-2">Experience the future of lottery analysis</p>
                </div>
            </div>

            <!-- Advanced Visualizations Grid -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <!-- Hot/Cold Analysis -->
                <div class="glass-card p-6 wow-effect">
                    <h3 class="text-xl font-bold mb-4">🔥 Hot/Cold Number Analysis</h3>
                    <div class="chart-container">
                        <div id="hot-cold-chart" class="h-96"></div>
                    </div>
                </div>
                
                <!-- Number Pair Network -->
                <div class="glass-card p-6 wow-effect">
                    <h3 class="text-xl font-bold mb-4">🕸️ Number Pair Network</h3>
                    <div class="chart-container">
                        <div id="network-chart" class="h-96"></div>
                    </div>
                </div>
            </div>

            <!-- Gap Analysis and Probability Zones -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <!-- Draw Gap Analysis -->
                <div class="glass-card p-6 wow-effect">
                    <h3 class="text-xl font-bold mb-4">⏰ Overdue Number Analysis</h3>
                    <div class="chart-container">
                        <div id="gap-analysis-chart" class="h-96"></div>
                    </div>
                </div>
                
                <!-- Probability Zones -->
                <div class="glass-card p-6 wow-effect">
                    <h3 class="text-xl font-bold mb-4">🎯 Sum & Range Probability Zones</h3>
                    <div class="chart-container">
                        <div id="probability-zones-chart" class="h-96"></div>
                    </div>
                </div>
            </div>

            <!-- Statistics Dashboard -->
            <div class="glass-card p-6 wow-effect">
                <h3 class="text-xl font-bold mb-4">📋 Real-time Analytics Dashboard</h3>
                <div id="statistics-container" class="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <!-- Statistics will be populated by JavaScript -->
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentLotteryType = 'powerball';
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeLotteryTabs();
            loadAnalytics();
            refreshSystemStatus();
        });
        
        function initializeLotteryTabs() {
            const tabs = document.querySelectorAll('.lottery-tab');
            tabs.forEach(tab => {
                tab.addEventListener('click', function() {
                    // Remove active class from all tabs
                    tabs.forEach(t => t.classList.remove('active'));
                    // Add active class to clicked tab
                    this.classList.add('active');
                    
                    // Update current lottery type
                    currentLotteryType = this.dataset.lottery;
                    loadAnalytics();
                });
            });
        }
        
        async function generatePredictions() {
            const nPredictions = document.getElementById('n-predictions').value;
            const container = document.getElementById('predictions-container');
            
            // Show loading with enhanced animation
            container.innerHTML = `
                <div class="loading">
                    <div class="flex justify-center mb-4">
                        <div class="prediction-ball">
                            <i class="fas fa-spinner fa-spin"></i>
                        </div>
                    </div>
                    <p class="text-lg">🤖 AI is analyzing ${currentLotteryType.replace('_', ' ')} patterns...</p>
                    <p class="text-sm text-gray-400 mt-2">Processing 10 mathematical pillars + LLM reasoning</p>
                </div>
            `;
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        lottery_type: currentLotteryType,
                        n_predictions: parseInt(nPredictions)
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayPredictions(data.predictions, data.lottery_config);
                } else {
                    container.innerHTML = `<p class="text-red-400">Error: ${data.error}</p>`;
                }
            } catch (error) {
                container.innerHTML = `<p class="text-red-400">Error: ${error.message}</p>`;
            }
        }
        
        function displayPredictions(predictions, config) {
            const container = document.getElementById('predictions-container');
            let html = '';
            
            predictions.forEach((pred, index) => {
                const isPickGame = ['pick_3', 'pick_4', 'pick_5'].includes(currentLotteryType);
                
                html += `
                    <div class="mb-6 p-6 bg-gradient-to-r from-white/5 to-white/10 rounded-xl wow-effect">
                        <h4 class="text-xl font-bold mb-4 gradient-text">
                            🎯 Prediction ${index + 1} - ${config.name}
                        </h4>
                        <div class="flex justify-center items-center space-x-4 mb-6">
                            ${pred.numbers.map((num, idx) => `
                                <div class="prediction-ball ${isPickGame ? 'text-2xl' : ''}" 
                                     style="animation-delay: ${idx * 0.1}s">
                                    ${num}
                                </div>
                            `).join('')}
                            ${pred.bonus_number ? `
                                <div class="prediction-ball bonus-ball" style="animation-delay: ${pred.numbers.length * 0.1}s">
                                    ${pred.bonus_number}
                                </div>
                            ` : ''}
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                            <div class="text-center">
                                <div class="text-sm text-gray-400">AI Confidence</div>
                                <div class="text-2xl font-bold text-green-400">${(pred.confidence * 100).toFixed(1)}%</div>
                            </div>
                            <div class="text-center">
                                <div class="text-sm text-gray-400">Pillars Used</div>
                                <div class="text-2xl font-bold text-blue-400">${Object.keys(pred.pillar_contributions).length}</div>
                            </div>
                            <div class="text-center">
                                <div class="text-sm text-gray-400">Analysis Type</div>
                                <div class="text-lg font-bold text-purple-400">Hybrid AI</div>
                            </div>
                        </div>
                        <div class="bg-white/5 p-4 rounded-lg mb-3">
                            <div class="text-sm font-semibold text-gray-300 mb-2">🧠 AI Reasoning:</div>
                            <div class="text-sm text-gray-300">${pred.reasoning}</div>
                        </div>
                        <div class="text-xs text-gray-400 text-center">
                            Generated: ${new Date(pred.timestamp).toLocaleString()}
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        async function loadAnalytics() {
            try {
                const response = await fetch(`/api/enhanced_analytics/${currentLotteryType}`);
                const data = await response.json();
                
                if (data.success) {
                    // Update all meaningful charts
                    if (data.charts.hot_cold_analysis) {
                        Plotly.newPlot('hot-cold-chart', data.charts.hot_cold_analysis.data, data.charts.hot_cold_analysis.layout, {responsive: true});
                    }
                    if (data.charts.network_analysis) {
                        Plotly.newPlot('network-chart', data.charts.network_analysis.data, data.charts.network_analysis.layout, {responsive: true});
                    }
                    if (data.charts.gap_analysis) {
                        Plotly.newPlot('gap-analysis-chart', data.charts.gap_analysis.data, data.charts.gap_analysis.layout, {responsive: true});
                    }
                    if (data.charts.probability_zones) {
                        Plotly.newPlot('probability-zones-chart', data.charts.probability_zones.data, data.charts.probability_zones.layout, {responsive: true});
                    }
                    
                    // Update statistics
                    updateStatistics(data.statistics);
                }
            } catch (error) {
                console.error('Analytics loading error:', error);
            }
        }
        
        function updateStatistics(stats) {
            const container = document.getElementById('statistics-container');
            
            let overdueInfo = '';
            if (stats.overdue_numbers && stats.overdue_numbers.length > 0) {
                const topOverdue = stats.overdue_numbers[0];
                overdueInfo = `
                    <div class="text-center p-4 bg-white/5 rounded-lg">
                        <div class="text-3xl font-bold text-orange-400">${topOverdue.number}</div>
                        <div class="text-sm text-gray-400">Most Overdue</div>
                        <div class="text-xs text-gray-500">${topOverdue.gap} draws</div>
                    </div>
                `;
            }
            
            container.innerHTML = `
                <div class="text-center p-4 bg-white/5 rounded-lg">
                    <div class="text-3xl font-bold text-blue-400">${stats.total_draws}</div>
                    <div class="text-sm text-gray-400">Total Draws</div>
                </div>
                <div class="text-center p-4 bg-white/5 rounded-lg">
                    <div class="text-3xl font-bold text-green-400">${stats.total_predictions}</div>
                    <div class="text-sm text-gray-400">AI Predictions</div>
                </div>
                <div class="text-center p-4 bg-white/5 rounded-lg">
                    <div class="text-3xl font-bold text-purple-400">${stats.avg_sum ? stats.avg_sum.toFixed(0) : 'N/A'}</div>
                    <div class="text-sm text-gray-400">Avg Sum</div>
                </div>
                <div class="text-center p-4 bg-white/5 rounded-lg">
                    <div class="text-3xl font-bold text-yellow-400">${stats.avg_range ? stats.avg_range.toFixed(0) : 'N/A'}</div>
                    <div class="text-sm text-gray-400">Avg Range</div>
                </div>
                <div class="text-center p-4 bg-white/5 rounded-lg">
                    <div class="text-3xl font-bold text-red-400">${stats.hot_numbers ? stats.hot_numbers.length : 0}</div>
                    <div class="text-sm text-gray-400">Hot Numbers</div>
                </div>
                ${overdueInfo}
            `;
        }
        
        async function refreshSystemStatus() {
            try {
                const response = await fetch('/api/system_status');
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('system-status').innerHTML = '🟢 Active';
                    document.getElementById('active-lotteries').innerHTML = `${data.status.supported_lotteries} Lottery Systems`;
                    
                    // Update pillar information
                    const pillarContainer = document.getElementById('pillar-info');
                    let html = '';
                    
                    data.status.pillars.forEach(pillar => {
                        html += `
                            <div class="pillar-contribution p-3 rounded-lg">
                                <div class="font-semibold">${pillar.name}</div>
                                <div class="text-sm text-gray-400">
                                    Weight: ${(pillar.weight * 100).toFixed(0)}% | 
                                    Performance: ${(pillar.avg_performance * 100).toFixed(1)}%
                                </div>
                            </div>
                        `;
                    });
                    
                    pillarContainer.innerHTML = html;
                }
            } catch (error) {
                document.getElementById('system-status').innerHTML = '🔴 Error';
            }
        }
        
        // Add some visual effects
        setInterval(() => {
            const balls = document.querySelectorAll('.prediction-ball');
            balls.forEach((ball, index) => {
                setTimeout(() => {
                    ball.style.transform = 'scale(1.1)';
                    setTimeout(() => {
                        ball.style.transform = 'scale(1)';
                    }, 200);
                }, index * 100);
            });
        }, 5000);
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_template)

# Enhanced Flask routes
@app.route('/')
def index():
    """Enhanced main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Enhanced prediction API with support for all lottery systems"""
    try:
        data = request.get_json()
        lottery_type = data.get('lottery_type', 'powerball')
        n_predictions = int(data.get('n_predictions', 1))
        
        if lottery_type not in lottery_data:
            return jsonify({'error': f'No data available for {lottery_type}. Please upload data first.'}), 400
        
        # Generate predictions
        predictions = enhanced_patternsight.predict(lottery_data[lottery_type], lottery_type, n_predictions)
        
        # Store in history
        if lottery_type not in prediction_history:
            prediction_history[lottery_type] = []
        prediction_history[lottery_type].extend(predictions)
        
        # Convert to JSON-serializable format
        result = []
        for pred in predictions:
            result.append({
                'numbers': [int(n) for n in pred.numbers],
                'bonus_number': int(pred.bonus_number) if pred.bonus_number else None,
                'confidence': float(pred.confidence),
                'pillar_contributions': {k: float(v) for k, v in pred.pillar_contributions.items()},
                'reasoning': pred.reasoning,
                'timestamp': pred.timestamp.isoformat()
            })
        
        return jsonify({
            'success': True,
            'predictions': result,
            'lottery_config': {
                'name': enhanced_patternsight.lottery_configs[lottery_type].name,
                'main_numbers': enhanced_patternsight.lottery_configs[lottery_type].main_numbers,
                'main_range': enhanced_patternsight.lottery_configs[lottery_type].main_range,
                'bonus_numbers': enhanced_patternsight.lottery_configs[lottery_type].bonus_numbers,
                'bonus_range': enhanced_patternsight.lottery_configs[lottery_type].bonus_range
            }
        })
        
    except Exception as e:
        logger.error(f"Enhanced prediction API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhanced_analytics/<lottery_type>')
def api_enhanced_analytics(lottery_type):
    """Enhanced analytics API with meaningful visualizations"""
    try:
        if lottery_type not in lottery_data:
            return jsonify({'error': f'No data available for {lottery_type}'}), 400
        
        data = lottery_data[lottery_type]
        config = enhanced_patternsight.lottery_configs[lottery_type]
        
        # Generate meaningful charts
        hot_cold_chart = create_hot_cold_trend_analysis(data, lottery_type)
        network_chart = create_number_pair_network(data, lottery_type)
        gap_analysis_chart = create_draw_gap_analysis(data, lottery_type)
        probability_zones_chart = create_sum_range_probability_zones(data, lottery_type)
        
        # Get prediction history for analysis
        predictions = prediction_history.get(lottery_type, [])
        
        # Calculate enhanced statistics
        if lottery_type in ['pick_3', 'pick_4', 'pick_5']:
            all_digits = [digit for _, row in data.iterrows() for digit in row['numbers']]
            from collections import Counter
            digit_counter = Counter(all_digits)
            
            # Calculate overdue digits
            last_seen = {}
            current_draw = len(data) - 1
            for idx, (_, row) in enumerate(data.iterrows()):
                for digit in row['numbers']:
                    last_seen[digit] = idx
            
            overdue_digits = []
            for digit in range(10):
                if digit in last_seen:
                    gap = current_draw - last_seen[digit]
                    overdue_digits.append((digit, gap))
            
            overdue_digits.sort(key=lambda x: x[1], reverse=True)
            
            stats = {
                'total_draws': len(data),
                'date_range': {
                    'start': data['date'].min().isoformat() if not data.empty else None,
                    'end': data['date'].max().isoformat() if not data.empty else None
                },
                'most_frequent': [{'number': digit, 'frequency': freq} for digit, freq in digit_counter.most_common(5)],
                'least_frequent': [{'number': digit, 'frequency': freq} for digit, freq in digit_counter.most_common()[-5:]],
                'total_predictions': len(predictions),
                'hot_numbers': [digit for digit, freq in digit_counter.most_common(3)],
                'overdue_numbers': [{'number': digit, 'gap': gap} for digit, gap in overdue_digits[:5]],
                'avg_sum': None,  # Not applicable for pick games
                'avg_range': None  # Not applicable for pick games
            }
        else:
            all_numbers = [num for _, row in data.iterrows() for num in row['numbers']]
            from collections import Counter
            freq_counter = Counter(all_numbers)
            
            # Calculate overdue numbers
            last_seen = {}
            current_draw = len(data) - 1
            for idx, (_, row) in enumerate(data.iterrows()):
                for num in row['numbers']:
                    last_seen[num] = idx
            
            overdue_numbers = []
            for num in range(config.main_range[0], config.main_range[1] + 1):
                if num in last_seen:
                    gap = current_draw - last_seen[num]
                    overdue_numbers.append((num, gap))
            
            overdue_numbers.sort(key=lambda x: x[1], reverse=True)
            
            stats = {
                'total_draws': len(data),
                'date_range': {
                    'start': data['date'].min().isoformat() if not data.empty else None,
                    'end': data['date'].max().isoformat() if not data.empty else None
                },
                'most_frequent': [{'number': num, 'frequency': freq} for num, freq in freq_counter.most_common(10)],
                'least_frequent': [{'number': num, 'frequency': freq} for num, freq in freq_counter.most_common()[-10:]],
                'avg_sum': float(np.mean([sum(row['numbers']) for _, row in data.iterrows()])),
                'avg_range': float(np.mean([max(row['numbers']) - min(row['numbers']) for _, row in data.iterrows()])),
                'total_predictions': len(predictions),
                'hot_numbers': [num for num, freq in freq_counter.most_common(5)],
                'overdue_numbers': [{'number': num, 'gap': gap} for num, gap in overdue_numbers[:10]]
            }
        
        return jsonify({
            'success': True,
            'charts': {
                'hot_cold_analysis': json.loads(hot_cold_chart) if hot_cold_chart != '{}' else None,
                'network_analysis': json.loads(network_chart) if network_chart != '{}' else None,
                'gap_analysis': json.loads(gap_analysis_chart) if gap_analysis_chart != '{}' else None,
                'probability_zones': json.loads(probability_zones_chart) if probability_zones_chart != '{}' else None
            },
            'statistics': stats,
            'config': {
                'name': config.name,
                'main_numbers': config.main_numbers,
                'main_range': config.main_range,
                'bonus_numbers': config.bonus_numbers,
                'bonus_range': config.bonus_range,
                'country': config.country
            }
        })
        
    except Exception as e:
        logger.error(f"Enhanced analytics API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_data', methods=['POST'])
def api_upload_data():
    """API endpoint to upload lottery data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        lottery_type = request.form.get('lottery_type', 'powerball')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = f"{lottery_type}_data.json"
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)
        
        # Load data
        data = load_lottery_data_from_file(filepath, lottery_type)
        
        if not data.empty:
            lottery_data[lottery_type] = data
            logger.info(f"✅ Uploaded and loaded {len(data)} draws for {lottery_type}")
            
            return jsonify({
                'success': True,
                'message': f'Successfully loaded {len(data)} draws for {lottery_type}',
                'draws_count': len(data)
            })
        else:
            return jsonify({'error': 'Failed to parse lottery data'}), 400
            
    except Exception as e:
        logger.error(f"Data upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_status')
def api_system_status():
    """Enhanced system status API"""
    try:
        status = {
            'system_version': 'PatternSight v4.0 Enhanced',
            'active_pillars': len(enhanced_patternsight.pillars),
            'supported_lotteries': len(enhanced_patternsight.lottery_configs),
            'loaded_datasets': len(lottery_data),
            'total_predictions': sum(len(predictions) for predictions in prediction_history.values()),
            'pillars': []
        }
        
        for pillar in enhanced_patternsight.pillars:
            pillar_info = {
                'name': pillar.name,
                'weight': pillar.weight,
                'avg_performance': pillar.get_average_performance(),
                'total_runs': len(pillar.performance_history)
            }
            status['pillars'].append(pillar_info)
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        logger.error(f"System status API error: {e}")
        return jsonify({'error': str(e)}), 500

def main():
    """Main function to run the enhanced dashboard"""
    logger.info("🚀 Starting PatternSight v4.0 Enhanced Dashboard...")
    
    # Load real lottery data
    logger.info("📊 Loading real lottery data from files...")
    global lottery_data
    lottery_data = load_real_lottery_data()
    
    # Update available lotteries
    available_lotteries = list(lottery_data.keys())
    logger.info(f"✅ Available lottery systems: {available_lotteries}")
    
    # Pre-load data summary
    for lottery_type, data in lottery_data.items():
        if not data.empty:
            date_range = f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}"
            logger.info(f"✅ {lottery_type}: {len(data)} draws ({date_range})")
    
    logger.info("✅ Enhanced Dashboard initialized successfully!")
    logger.info("🌐 Starting enhanced web server...")
    logger.info("📊 Ready for multiple lottery systems with advanced AI analytics!")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == "__main__":
    main()

