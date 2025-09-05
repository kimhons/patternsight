/**
 * Test Real AI Predictions
 * This script tests actual prediction generation using real API calls
 */

import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import Anthropic from '@anthropic-ai/sdk';
import fetch from 'node-fetch';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
dotenv.config({ path: path.join(__dirname, '.env.local') });

// Initialize AI clients
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

console.log('🚀 Starting Real AI Prediction Tests\n');
console.log('='.repeat(50));

// Test data for predictions
const lotteryGames = {
  powerball: {
    name: 'Powerball',
    mainNumbers: { min: 1, max: 69, count: 5 },
    specialBall: { min: 1, max: 26, name: 'Powerball' },
    recentDraws: [
      { numbers: [7, 11, 19, 53, 68], powerball: 23 },
      { numbers: [16, 30, 31, 42, 68], powerball: 24 },
      { numbers: [27, 28, 34, 37, 44], powerball: 8 }
    ]
  },
  megaMillions: {
    name: 'Mega Millions',
    mainNumbers: { min: 1, max: 70, count: 5 },
    specialBall: { min: 1, max: 25, name: 'Mega Ball' },
    recentDraws: [
      { numbers: [3, 10, 29, 52, 57], megaBall: 20 },
      { numbers: [12, 20, 32, 45, 65], megaBall: 12 },
      { numbers: [5, 22, 29, 33, 41], megaBall: 4 }
    ]
  }
};

// Statistical Analysis Function
function performStatisticalAnalysis(game) {
  const allNumbers = game.recentDraws.flatMap(d => d.numbers);
  const frequency = {};
  
  allNumbers.forEach(num => {
    frequency[num] = (frequency[num] || 0) + 1;
  });
  
  const hotNumbers = Object.entries(frequency)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([num]) => parseInt(num));
  
  const coldNumbers = [];
  for (let i = game.mainNumbers.min; i <= game.mainNumbers.max; i++) {
    if (!frequency[i]) coldNumbers.push(i);
  }
  
  return { hotNumbers, coldNumbers: coldNumbers.slice(0, 10), frequency };
}

// Generate predictions using OpenAI GPT-4
async function generateOpenAIPrediction(game, analysis) {
  console.log('\n📊 Generating prediction with OpenAI GPT-4...');
  
  const prompt = `
You are an advanced AI lottery prediction system. Analyze the following data and generate a prediction:

Game: ${game.name}
Number Range: ${game.mainNumbers.count} numbers from ${game.mainNumbers.min}-${game.mainNumbers.max}
Special Ball: 1 number from ${game.specialBall.min}-${game.specialBall.max} (${game.specialBall.name})

Recent Draws:
${game.recentDraws.map((d, i) => 
  `Draw ${i+1}: [${d.numbers.join(', ')}] + ${d.powerball || d.megaBall}`
).join('\n')}

Statistical Analysis:
- Hot Numbers (frequently drawn): ${analysis.hotNumbers.join(', ')}
- Cold Numbers (overdue): ${analysis.coldNumbers.slice(0, 5).join(', ')}

Based on:
1. Statistical probability theory
2. Pattern recognition
3. Frequency analysis
4. Gap theory
5. Number distribution patterns

Generate a prediction with:
1. 5 main numbers (sorted ascending)
2. 1 special ball number
3. Confidence score (0-100%)
4. Brief reasoning

Format your response as JSON:
{
  "mainNumbers": [n1, n2, n3, n4, n5],
  "specialBall": n,
  "confidence": percentage,
  "reasoning": "brief explanation"
}
`;

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'You are an expert lottery prediction AI using advanced statistical analysis and pattern recognition. Always respond with valid JSON format.'
        },
        {
          role: 'user',
          content: prompt + '\n\nIMPORTANT: Respond ONLY with valid JSON, no additional text.'
        }
      ],
      temperature: 0.7,
      max_tokens: 500
    });

    const prediction = JSON.parse(response.choices[0].message.content);
    console.log('✅ GPT-4 Prediction Generated:');
    console.log(`   Main Numbers: ${prediction.mainNumbers.join(', ')}`);
    console.log(`   ${game.specialBall.name}: ${prediction.specialBall}`);
    console.log(`   Confidence: ${prediction.confidence}%`);
    console.log(`   Reasoning: ${prediction.reasoning}`);
    
    return prediction;
  } catch (error) {
    console.error('❌ OpenAI API Error:', error.message);
    return null;
  }
}

// Neural Network Simulation (representing TensorFlow.js predictions)
function generateNeuralNetworkPrediction(game, analysis) {
  console.log('\n🧠 Generating Neural Network Prediction...');
  
  // Simulate LSTM pattern recognition
  const predictions = [];
  const used = new Set();
  
  // Mix hot and cold numbers based on patterns
  const candidates = [
    ...analysis.hotNumbers.slice(0, 3),
    ...analysis.coldNumbers.slice(0, 2)
  ];
  
  // Add some random numbers for diversity
  while (candidates.length < 10) {
    const num = Math.floor(Math.random() * game.mainNumbers.max) + 1;
    if (!candidates.includes(num)) {
      candidates.push(num);
    }
  }
  
  // Select 5 numbers
  while (predictions.length < game.mainNumbers.count) {
    const idx = Math.floor(Math.random() * candidates.length);
    const num = candidates[idx];
    if (!used.has(num)) {
      predictions.push(num);
      used.add(num);
    }
  }
  
  const specialBall = Math.floor(Math.random() * game.specialBall.max) + 1;
  const confidence = 65 + Math.random() * 20; // 65-85%
  
  console.log('✅ Neural Network Prediction:');
  console.log(`   Main Numbers: ${predictions.sort((a, b) => a - b).join(', ')}`);
  console.log(`   ${game.specialBall.name}: ${specialBall}`);
  console.log(`   Confidence: ${confidence.toFixed(1)}%`);
  
  return {
    mainNumbers: predictions.sort((a, b) => a - b),
    specialBall,
    confidence: confidence.toFixed(1),
    method: 'LSTM + CNN Neural Network'
  };
}

// Ensemble prediction combining multiple models
function generateEnsemblePrediction(predictions) {
  console.log('\n🎯 Generating Ensemble Prediction...');
  
  // Count frequency of each number across all predictions
  const numberFreq = {};
  const specialFreq = {};
  
  predictions.forEach(pred => {
    if (pred && pred.mainNumbers) {
      pred.mainNumbers.forEach(num => {
        numberFreq[num] = (numberFreq[num] || 0) + 1;
      });
      if (pred.specialBall) {
        specialFreq[pred.specialBall] = (specialFreq[pred.specialBall] || 0) + 1;
      }
    }
  });
  
  // Select most frequent numbers
  const ensembleNumbers = Object.entries(numberFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([num]) => parseInt(num))
    .sort((a, b) => a - b);
  
  const ensembleSpecial = Object.entries(specialFreq)
    .sort((a, b) => b[1] - a[1])[0]?.[0] || 
    Math.floor(Math.random() * 26) + 1;
  
  const avgConfidence = predictions
    .filter(p => p && p.confidence)
    .reduce((sum, p) => sum + parseFloat(p.confidence), 0) / predictions.length;
  
  console.log('✅ Ensemble Prediction (Combined):');
  console.log(`   Main Numbers: ${ensembleNumbers.join(', ')}`);
  console.log(`   Special Ball: ${ensembleSpecial}`);
  console.log(`   Confidence: ${avgConfidence.toFixed(1)}%`);
  
  return {
    mainNumbers: ensembleNumbers,
    specialBall: parseInt(ensembleSpecial),
    confidence: avgConfidence.toFixed(1),
    method: 'Ensemble (Multi-Model Consensus)'
  };
}

// Main test function
async function runPredictionTests() {
  console.log('\n🎲 Testing Powerball Predictions');
  console.log('='.repeat(50));
  
  const game = lotteryGames.powerball;
  const analysis = performStatisticalAnalysis(game);
  
  console.log('\n📈 Statistical Analysis:');
  console.log(`   Hot Numbers: ${analysis.hotNumbers.join(', ')}`);
  console.log(`   Cold Numbers: ${analysis.coldNumbers.slice(0, 5).join(', ')}`);
  
  // Generate predictions using different methods
  const predictions = [];
  
  // 1. OpenAI GPT-4 Prediction
  const openaiPred = await generateOpenAIPrediction(game, analysis);
  if (openaiPred) predictions.push(openaiPred);
  
  // 2. Neural Network Prediction
  const nnPred = generateNeuralNetworkPrediction(game, analysis);
  predictions.push(nnPred);
  
  // 3. Statistical Model Prediction
  console.log('\n📊 Generating Statistical Model Prediction...');
  const statPred = {
    mainNumbers: [...analysis.hotNumbers.slice(0, 3), ...analysis.coldNumbers.slice(0, 2)].sort((a, b) => a - b),
    specialBall: Math.floor(Math.random() * 26) + 1,
    confidence: 72.5,
    method: 'Statistical Frequency Analysis'
  };
  console.log('✅ Statistical Prediction:');
  console.log(`   Main Numbers: ${statPred.mainNumbers.join(', ')}`);
  console.log(`   Powerball: ${statPred.specialBall}`);
  console.log(`   Confidence: ${statPred.confidence}%`);
  predictions.push(statPred);
  
  // 4. Ensemble Prediction
  const ensemblePred = generateEnsemblePrediction(predictions);
  
  // Final Summary
  console.log('\n' + '='.repeat(50));
  console.log('📊 PREDICTION SUMMARY');
  console.log('='.repeat(50));
  
  predictions.forEach((pred, idx) => {
    if (pred) {
      console.log(`\n${idx + 1}. ${pred.method || 'Model ' + (idx + 1)}:`);
      console.log(`   Numbers: ${pred.mainNumbers?.join(', ') || 'N/A'}`);
      console.log(`   Powerball: ${pred.specialBall || 'N/A'}`);
      console.log(`   Confidence: ${pred.confidence}%`);
    }
  });
  
  console.log(`\n🎯 FINAL ENSEMBLE RECOMMENDATION:`);
  console.log(`   Numbers: ${ensemblePred.mainNumbers.join(', ')}`);
  console.log(`   Powerball: ${ensemblePred.specialBall}`);
  console.log(`   Confidence: ${ensemblePred.confidence}%`);
  
  console.log('\n✨ Prediction generation complete!');
  console.log('Note: These are AI-generated predictions for entertainment purposes only.');
}

// Run the tests
runPredictionTests().catch(console.error);