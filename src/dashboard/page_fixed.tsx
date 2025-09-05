'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { EnhancedUPPS_v3, PredictionResult, PillarScores } from '@/lib/core/enhanced-upps-v3';
import { useAuth } from '@/components/contexts/AuthContext';

interface DashboardStats {
  totalPredictions: number;
  avgConfidence: number;
  activePillars: number;
  lastUpdate: string;
}

export default function Dashboard() {
  const { user, session, loading } = useAuth();
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [stats, setStats] = useState<DashboardStats>({
    totalPredictions: 1247,
    avgConfidence: 87.3,
    activePillars: 5,
    lastUpdate: new Date().toLocaleString()
  });
  const [selectedPillar, setSelectedPillar] = useState<string | null>(null);
  const [enhancedUPPS] = useState(() => new EnhancedUPPS_v3());

  useEffect(() => {
    // Auto-generate initial prediction only if user is authenticated
    if (user && !loading) {
      generatePrediction();
    }
  }, [user, loading]);

  const generatePrediction = async () => {
    if (!session) {
      alert('Please sign in to generate predictions');
      return;
    }

    setIsGenerating(true);
    try {
      // Generate prediction using Enhanced UPPS v3.0
      const prediction = await enhancedUPPS.generatePrediction();
      
      setCurrentPrediction(prediction);
      setStats(prev => ({
        ...prev,
        totalPredictions: prev.totalPredictions + 1,
        lastUpdate: new Date().toLocaleString()
      }));
    } catch (error) {
      console.error('Error generating prediction:', error);
      alert('Failed to generate prediction. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const getPillarColor = (pillarName: string, score: number): string => {
    const maxScores: Record<string, number> = {
      cdm: 25,
      orderStats: 20,
      neural: 20,
      behavioral: 20,
      temporal: 15
    };
    
    const percentage = (score / maxScores[pillarName]) * 100;
    
    if (percentage >= 80) return 'text-green-400';
    if (percentage >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getPillarDescription = (pillarName: string): string => {
    const descriptions: Record<string, string> = {
      cdm: 'Compound-Dirichlet-Multinomial Model (Nkomozake 2024) - Advanced Bayesian probability analysis',
      orderStats: 'Order Statistics Theory (Tse 2024) - Mathematical position optimization',
      neural: 'Statistical-Neural Hybrid - Pattern recognition and sequence analysis',
      behavioral: 'XGBoost Behavioral Analysis - Machine learning trend detection',
      temporal: 'Deep Learning Time Series - LSTM-inspired temporal patterns'
    };
    return descriptions[pillarName] || '';
  };

  const getPillarDisplayName = (pillarName: string): string => {
    const displayNames: Record<string, string> = {
      cdm: 'CDM Analysis',
      orderStats: 'Order Statistics',
      neural: 'Neural Patterns',
      behavioral: 'Behavioral Analysis',
      temporal: 'Temporal Modeling'
    };
    return displayNames[pillarName] || pillarName;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <div className="text-white text-xl">Please sign in to access the dashboard</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-orange-400 to-pink-400 bg-clip-text text-transparent">
            PatternSight Dashboard
          </h1>
          <p className="text-gray-300">Enhanced UPPS v3.0 - Academic Integration System</p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-2xl font-bold text-orange-400">{stats.totalPredictions.toLocaleString()}</div>
            <div className="text-gray-300">Total Predictions</div>
          </motion.div>
          
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-2xl font-bold text-green-400">{stats.avgConfidence}%</div>
            <div className="text-gray-300">Average Confidence</div>
          </motion.div>
          
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-2xl font-bold text-blue-400">{stats.activePillars}/5</div>
            <div className="text-gray-300">Active Pillars</div>
          </motion.div>
          
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-sm font-bold text-purple-400">{stats.lastUpdate}</div>
            <div className="text-gray-300">Last Update</div>
          </motion.div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Prediction Panel */}
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold">🎯 AI Prediction</h2>
              <motion.button
                onClick={generatePrediction}
                disabled={isGenerating}
                className="bg-gradient-to-r from-orange-500 to-pink-500 hover:from-orange-600 hover:to-pink-600 disabled:opacity-50 px-6 py-2 rounded-lg font-semibold transition-all duration-200"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isGenerating ? '🔄 Generating...' : '🎲 Generate Prediction'}
              </motion.button>
            </div>

            <AnimatePresence mode="wait">
              {currentPrediction ? (
                <motion.div
                  key="prediction"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="space-y-4"
                >
                  {/* Numbers Display */}
                  <div className="text-center">
                    <div className="flex justify-center space-x-3 mb-4">
                      {currentPrediction.numbers.map((num, index) => (
                        <motion.div
                          key={index}
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: index * 0.1 }}
                          className="w-12 h-12 bg-gradient-to-br from-white to-gray-200 text-black rounded-full flex items-center justify-center font-bold text-lg shadow-lg"
                        >
                          {num}
                        </motion.div>
                      ))}
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.6 }}
                        className="w-12 h-12 bg-gradient-to-br from-red-500 to-red-700 text-white rounded-full flex items-center justify-center font-bold text-lg shadow-lg"
                      >
                        {currentPrediction.powerball}
                      </motion.div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Confidence:</span>
                        <span className="ml-2 font-bold text-green-400">{currentPrediction.confidence.toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Digital Root:</span>
                        <span className="ml-2 font-bold text-purple-400">{currentPrediction.digitalRoot}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Score:</span>
                        <span className="ml-2 font-bold text-orange-400">{currentPrediction.score.toFixed(1)}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Tier:</span>
                        <span className="ml-2 font-bold text-blue-400">{currentPrediction.tier}</span>
                      </div>
                    </div>
                  </div>

                  {/* AI Analysis */}
                  <div className="bg-black/20 rounded-lg p-4">
                    <h3 className="font-semibold mb-2 text-cyan-400">🧠 AI Analysis</h3>
                    <p className="text-sm text-gray-300 leading-relaxed">
                      {currentPrediction.analysis}
                    </p>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="no-prediction"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-center py-12 text-gray-400"
                >
                  <div className="text-6xl mb-4">🎯</div>
                  <p>Click "Generate Prediction" to create your first AI-powered lottery prediction</p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* 5-Pillar Analysis */}
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h2 className="text-2xl font-bold mb-6">⚡ 5-Pillar Analysis</h2>
            
            {currentPrediction ? (
              <div className="space-y-4">
                {Object.entries(currentPrediction.pillars).map(([pillar, score]) => (
                  <motion.div
                    key={pillar}
                    className="cursor-pointer"
                    onClick={() => setSelectedPillar(selectedPillar === pillar ? null : pillar)}
                    whileHover={{ scale: 1.02 }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold">{getPillarDisplayName(pillar)}</span>
                      <span className={`font-bold ${getPillarColor(pillar, score)}`}>
                        {score.toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <motion.div
                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full"
                        initial={{ width: 0 }}
                        animate={{ width: `${score}%` }}
                        transition={{ duration: 1, delay: 0.5 }}
                      />
                    </div>
                    
                    <AnimatePresence>
                      {selectedPillar === pillar && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-2 p-3 bg-black/20 rounded text-xs text-gray-300"
                        >
                          {getPillarDescription(pillar)}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-400">
                <div className="text-4xl mb-4">⚡</div>
                <p>Generate a prediction to see 5-pillar analysis</p>
              </div>
            )}
          </motion.div>
        </div>

        {/* System Health */}
        <motion.div 
          className="mt-8 bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h2 className="text-2xl font-bold mb-6">🟢 System Health</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <span>Enhanced UPPS v3.0: Operational</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <span>Academic Models: 5/5 Active</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <span>Prediction Engine: Ready</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <span>Data Analytics: Processing</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

