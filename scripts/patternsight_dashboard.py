#!/usr/bin/env python3
"""
PatternSight v4.0 Interactive Web Dashboard
The Ultimate Multi-Lottery Prediction Platform Interface

Features:
- Real-time predictions across multiple lottery systems
- Interactive visualizations and analytics
- AI reasoning transparency
- Performance tracking and metrics
- Multi-lottery support with beautiful UI

Professor [Name], Ph.D. (MIT), Ph.D. (Harvard)
Computational and Mathematical Sciences Research Institute
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
from patternsight_unified_system import PatternSightV4Unified, PredictionResult
import os
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'patternsight_v4_dashboard_2025'

# Initialize PatternSight system
patternsight = PatternSightV4Unified()

# Global data storage
lottery_data = {}
prediction_history = {}

def load_lottery_datasets():
    """Load all available lottery datasets"""
    global lottery_data
    
    # Load Powerball data
    try:
        powerball_data = patternsight.load_lottery_data('/home/ubuntu/upload/powerball_data_5years.json', 'powerball')
        lottery_data['powerball'] = powerball_data
        logger.info(f"✅ Loaded {len(powerball_data)} Powerball draws")
    except Exception as e:
        logger.error(f"❌ Failed to load Powerball data: {e}")
    
    # Generate sample data for other lotteries (for demonstration)
    for lottery_type in ['mega_millions', 'euromillions', 'uk_lotto', 'canada_lotto']:
        try:
            sample_data = generate_sample_lottery_data(lottery_type, 500)
            lottery_data[lottery_type] = sample_data
            logger.info(f"✅ Generated {len(sample_data)} sample draws for {lottery_type}")
        except Exception as e:
            logger.error(f"❌ Failed to generate sample data for {lottery_type}: {e}")

def generate_sample_lottery_data(lottery_type: str, n_draws: int) -> pd.DataFrame:
    """Generate sample lottery data for demonstration"""
    config = patternsight.lottery_configs[lottery_type]
    draws = []
    
    start_date = datetime.now() - timedelta(days=n_draws * 3)
    
    for i in range(n_draws):
        draw_date = start_date + timedelta(days=i * 3)
        numbers = sorted(np.random.choice(range(config.main_range[0], config.main_range[1] + 1), 
                                        size=config.main_numbers, replace=False))
        bonus = np.random.randint(config.bonus_range[0], config.bonus_range[1] + 1) if config.bonus_numbers > 0 else None
        
        draw = {
            'date': draw_date,
            'numbers': numbers,
            'day_of_week': draw_date.strftime('%A'),
            'month': draw_date.month,
            'year': draw_date.year
        }
        
        if bonus:
            draw['bonus'] = bonus
        
        draws.append(draw)
    
    return pd.DataFrame(draws)

def create_frequency_chart(data: pd.DataFrame, lottery_type: str) -> str:
    """Create frequency analysis chart"""
    config = patternsight.lottery_configs[lottery_type]
    
    # Count number frequencies
    all_numbers = [num for _, row in data.iterrows() for num in row['numbers']]
    from collections import Counter
    freq_counter = Counter(all_numbers)
    
    numbers = list(range(config.main_range[0], config.main_range[1] + 1))
    frequencies = [freq_counter.get(num, 0) for num in numbers]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=numbers,
            y=frequencies,
            marker=dict(
                color=frequencies,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Frequency")
            ),
            hovertemplate='Number: %{x}<br>Frequency: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'{config.name} - Number Frequency Analysis',
        xaxis_title='Numbers',
        yaxis_title='Frequency',
        template='plotly_dark',
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_pattern_analysis_chart(data: pd.DataFrame, lottery_type: str) -> str:
    """Create pattern analysis visualization"""
    config = patternsight.lottery_configs[lottery_type]
    
    # Analyze sum patterns over time
    dates = []
    sums = []
    ranges = []
    
    for _, row in data.tail(100).iterrows():  # Last 100 draws
        dates.append(row['date'])
        sums.append(sum(row['numbers']))
        ranges.append(max(row['numbers']) - min(row['numbers']))
    
    # Create subplot
    fig = go.Figure()
    
    # Sum trend
    fig.add_trace(go.Scatter(
        x=dates,
        y=sums,
        mode='lines+markers',
        name='Sum Trend',
        line=dict(color='#00ff88', width=2),
        hovertemplate='Date: %{x}<br>Sum: %{y}<extra></extra>'
    ))
    
    # Range trend (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=dates,
        y=ranges,
        mode='lines+markers',
        name='Range Trend',
        yaxis='y2',
        line=dict(color='#ff6b6b', width=2),
        hovertemplate='Date: %{x}<br>Range: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{config.name} - Pattern Analysis (Last 100 Draws)',
        xaxis_title='Date',
        yaxis=dict(title='Sum', side='left', color='#00ff88'),
        yaxis2=dict(title='Range', side='right', overlaying='y', color='#ff6b6b'),
        template='plotly_dark',
        height=400,
        legend=dict(x=0.02, y=0.98)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_prediction_confidence_chart(predictions: List[PredictionResult]) -> str:
    """Create prediction confidence visualization"""
    if not predictions:
        return json.dumps({})
    
    prediction_nums = list(range(1, len(predictions) + 1))
    confidences = [pred.confidence * 100 for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=prediction_nums,
            y=confidences,
            marker=dict(
                color=confidences,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Confidence %")
            ),
            hovertemplate='Prediction: %{x}<br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Prediction Confidence Levels',
        xaxis_title='Prediction Number',
        yaxis_title='Confidence (%)',
        template='plotly_dark',
        height=300
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', 
                         lottery_systems=list(patternsight.lottery_configs.keys()),
                         pillars=[pillar.name for pillar in patternsight.pillars])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Generate predictions via API"""
    try:
        data = request.get_json()
        lottery_type = data.get('lottery_type', 'powerball')
        n_predictions = int(data.get('n_predictions', 1))
        
        if lottery_type not in lottery_data:
            return jsonify({'error': f'No data available for {lottery_type}'}), 400
        
        # Generate predictions
        predictions = patternsight.predict(lottery_data[lottery_type], lottery_type, n_predictions)
        
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
                'name': patternsight.lottery_configs[lottery_type].name,
                'main_numbers': patternsight.lottery_configs[lottery_type].main_numbers,
                'main_range': patternsight.lottery_configs[lottery_type].main_range,
                'bonus_numbers': patternsight.lottery_configs[lottery_type].bonus_numbers,
                'bonus_range': patternsight.lottery_configs[lottery_type].bonus_range
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/<lottery_type>')
def api_analytics(lottery_type):
    """Get analytics data for lottery system"""
    try:
        if lottery_type not in lottery_data:
            return jsonify({'error': f'No data available for {lottery_type}'}), 400
        
        data = lottery_data[lottery_type]
        config = patternsight.lottery_configs[lottery_type]
        
        # Generate charts
        frequency_chart = create_frequency_chart(data, lottery_type)
        pattern_chart = create_pattern_analysis_chart(data, lottery_type)
        
        # Get prediction history for confidence chart
        predictions = prediction_history.get(lottery_type, [])
        confidence_chart = create_prediction_confidence_chart(predictions[-20:])  # Last 20 predictions
        
        # Calculate statistics
        all_numbers = [num for _, row in data.iterrows() for num in row['numbers']]
        from collections import Counter
        freq_counter = Counter(all_numbers)
        
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
            'total_predictions': len(predictions)
        }
        
        return jsonify({
            'success': True,
            'charts': {
                'frequency': json.loads(frequency_chart),
                'patterns': json.loads(pattern_chart),
                'confidence': json.loads(confidence_chart) if confidence_chart != '{}' else None
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
        logger.error(f"Analytics API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_status')
def api_system_status():
    """Get system status and performance metrics"""
    try:
        status = {
            'system_version': 'PatternSight v4.0',
            'active_pillars': len(patternsight.pillars),
            'supported_lotteries': len(patternsight.lottery_configs),
            'loaded_datasets': len(lottery_data),
            'total_predictions': sum(len(predictions) for predictions in prediction_history.values()),
            'pillars': []
        }
        
        for pillar in patternsight.pillars:
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

# Create templates directory and HTML template
def create_dashboard_template():
    """Create the HTML template for the dashboard"""
    os.makedirs('templates', exist_ok=True)
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PatternSight v4.0 - Multi-Lottery Prediction Dashboard</title>
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
        }
        .loading {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        .pillar-contribution {
            background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid #667eea;
        }
    </style>
</head>
<body class="min-h-screen text-white">
    <!-- Header -->
    <header class="glass-card m-6 p-6">
        <div class="flex items-center justify-between">
            <div>
                <h1 class="text-4xl font-bold gradient-text">PatternSight v4.0</h1>
                <p class="text-gray-300 mt-2">The World's Most Advanced Multi-Lottery Prediction Platform</p>
            </div>
            <div class="flex items-center space-x-4">
                <div class="text-right">
                    <div class="text-sm text-gray-400">System Status</div>
                    <div class="text-green-400 font-semibold" id="system-status">🟢 Active</div>
                </div>
                <button onclick="refreshSystemStatus()" class="glass-card p-3 hover:bg-white/10 transition-all">
                    <i class="fas fa-sync-alt"></i>
                </button>
            </div>
        </div>
    </header>

    <!-- Main Dashboard -->
    <div class="mx-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Control Panel -->
        <div class="lg:col-span-1">
            <div class="glass-card p-6 mb-6">
                <h2 class="text-2xl font-bold mb-4">🎯 Prediction Control</h2>
                
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">Lottery System</label>
                    <select id="lottery-select" class="w-full bg-white/10 border border-white/20 rounded-lg p-3 text-white">
                        {% for lottery in lottery_systems %}
                        <option value="{{ lottery }}">{{ lottery.replace('_', ' ').title() }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="mb-4">
                    <label class="block text-sm font-medium mb-2">Number of Predictions</label>
                    <input type="number" id="n-predictions" min="1" max="10" value="1" 
                           class="w-full bg-white/10 border border-white/20 rounded-lg p-3 text-white">
                </div>
                
                <button onclick="generatePredictions()" 
                        class="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 
                               text-white font-bold py-3 px-6 rounded-lg transition-all transform hover:scale-105">
                    <i class="fas fa-magic mr-2"></i>Generate Predictions
                </button>
            </div>

            <!-- System Information -->
            <div class="glass-card p-6">
                <h3 class="text-xl font-bold mb-4">🏗️ System Architecture</h3>
                <div id="pillar-info" class="space-y-2">
                    {% for pillar in pillars %}
                    <div class="pillar-contribution p-3 rounded-lg">
                        <div class="font-semibold">{{ pillar }}</div>
                        <div class="text-sm text-gray-400">Active & Ready</div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>

        <!-- Results Panel -->
        <div class="lg:col-span-2">
            <!-- Predictions Display -->
            <div class="glass-card p-6 mb-6">
                <h2 class="text-2xl font-bold mb-4">🔮 Latest Predictions</h2>
                <div id="predictions-container" class="text-center text-gray-400 py-8">
                    <i class="fas fa-crystal-ball text-4xl mb-4"></i>
                    <p>Generate predictions to see AI-powered lottery analysis</p>
                </div>
            </div>

            <!-- Analytics Charts -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="glass-card p-6">
                    <h3 class="text-xl font-bold mb-4">📊 Frequency Analysis</h3>
                    <div id="frequency-chart" class="h-64"></div>
                </div>
                
                <div class="glass-card p-6">
                    <h3 class="text-xl font-bold mb-4">📈 Pattern Analysis</h3>
                    <div id="pattern-chart" class="h-64"></div>
                </div>
            </div>

            <!-- Statistics -->
            <div class="glass-card p-6 mt-6">
                <h3 class="text-xl font-bold mb-4">📋 System Statistics</h3>
                <div id="statistics-container" class="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <!-- Statistics will be populated by JavaScript -->
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentLotteryType = 'powerball';
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadAnalytics();
            refreshSystemStatus();
        });
        
        // Update lottery selection
        document.getElementById('lottery-select').addEventListener('change', function() {
            currentLotteryType = this.value;
            loadAnalytics();
        });
        
        async function generatePredictions() {
            const nPredictions = document.getElementById('n-predictions').value;
            const container = document.getElementById('predictions-container');
            
            // Show loading
            container.innerHTML = `
                <div class="loading">
                    <i class="fas fa-spinner fa-spin text-4xl mb-4"></i>
                    <p>AI is analyzing patterns and generating predictions...</p>
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
                html += `
                    <div class="mb-6 p-4 bg-white/5 rounded-lg">
                        <h4 class="text-lg font-bold mb-3">Prediction ${index + 1} - ${config.name}</h4>
                        <div class="flex justify-center items-center space-x-3 mb-4">
                            ${pred.numbers.map(num => `<div class="prediction-ball">${num}</div>`).join('')}
                            ${pred.bonus_number ? `<div class="prediction-ball bg-red-500">${pred.bonus_number}</div>` : ''}
                        </div>
                        <div class="text-center mb-3">
                            <span class="text-sm text-gray-400">Confidence: </span>
                            <span class="font-bold text-green-400">${(pred.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div class="text-sm text-gray-300 bg-white/5 p-3 rounded">
                            <strong>AI Reasoning:</strong> ${pred.reasoning}
                        </div>
                        <div class="mt-3 text-xs text-gray-400">
                            Generated: ${new Date(pred.timestamp).toLocaleString()}
                        </div>
                    </div>
                `;
            });
            
            container.innerHTML = html;
        }
        
        async function loadAnalytics() {
            try {
                const response = await fetch(`/api/analytics/${currentLotteryType}`);
                const data = await response.json();
                
                if (data.success) {
                    // Update charts
                    if (data.charts.frequency) {
                        Plotly.newPlot('frequency-chart', data.charts.frequency.data, data.charts.frequency.layout, {responsive: true});
                    }
                    if (data.charts.patterns) {
                        Plotly.newPlot('pattern-chart', data.charts.patterns.data, data.charts.patterns.layout, {responsive: true});
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
            container.innerHTML = `
                <div class="text-center">
                    <div class="text-2xl font-bold text-blue-400">${stats.total_draws}</div>
                    <div class="text-sm text-gray-400">Total Draws</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-green-400">${stats.total_predictions}</div>
                    <div class="text-sm text-gray-400">Predictions Made</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-purple-400">${stats.avg_sum.toFixed(0)}</div>
                    <div class="text-sm text-gray-400">Avg Sum</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-yellow-400">${stats.avg_range.toFixed(0)}</div>
                    <div class="text-sm text-gray-400">Avg Range</div>
                </div>
            `;
        }
        
        async function refreshSystemStatus() {
            try {
                const response = await fetch('/api/system_status');
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('system-status').innerHTML = '🟢 Active';
                    
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
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_template)

def main():
    """Main function to run the dashboard"""
    logger.info("🚀 Starting PatternSight v4.0 Dashboard...")
    
    # Load lottery datasets
    load_lottery_datasets()
    
    # Create dashboard template
    create_dashboard_template()
    
    logger.info("✅ Dashboard initialized successfully!")
    logger.info("🌐 Starting web server...")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()

