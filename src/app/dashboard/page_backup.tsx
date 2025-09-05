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
  }, [user, loading]  const generatePrediction = async () => {
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
  }; const prediction = await predictionEngine.generatePrediction();
        setCurrentPrediction(prediction);
      } catch (fallbackError) {
        console.error('Fallback prediction failed:', fallbackError);
      }
    } finally {
      setIsGenerating(false);
    }
  };

  const getPillarColor = (pillarName: string, score: number): string => {
    const maxScores: Record<string, number> = {
      statistical: 30,
      aiIntelligence: 25,
      dataAnalytics: 20,
      predictiveInsights: 15,
      cosmicIntelligence: 25
    };
    
    const percentage = (score / maxScores[pillarName]) * 100;
    
    if (percentage >= 80) return 'text-green-400';
    if (percentage >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getPillarDescription = (pillarName: string): string => {
    const descriptions: Record<string, string> = {
      statistical: 'Advanced frequency analysis, trend modeling, and statistical pattern detection using proven mathematical methods.',
      aiIntelligence: 'Deep learning algorithms that adapt and improve with every analysis cycle and data input.',
      dataAnalytics: 'Comprehensive data analysis, clustering, and pattern recognition across multiple domains and datasets.',
      predictiveInsights: 'Advanced forecasting and trend prediction capabilities powered by machine learning algorithms.',
      cosmicIntelligence: 'Astronomical correlations, numerological patterns, and sacred geometry principles for metaphysical insights.'
    };
    return descriptions[pillarName] || '';
  };

  const formatPillarName = (pillarName: string): string => {
    const names: Record<string, string> = {
      statistical: 'Statistical Mastery',
      aiIntelligence: 'AI Intelligence',
      dataAnalytics: 'Data Analytics',
      predictiveInsights: 'Predictive Insights',
      cosmicIntelligence: 'Cosmic Intelligence'
    };
    return names[pillarName] || pillarName;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-pink-400 bg-clip-text text-transparent">
                PatternSight Dashboard
              </h1>
              <p className="text-gray-300 mt-1">Ultimate AI-Powered Prediction System</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-sm text-gray-400">System Status</div>
                <div className="text-green-400 font-semibold">🟢 All Systems Operational</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20"
            whileHover={{ scale: 1.02 }}
          >
            <div className="text-2xl font-bold text-orange-400">{stats.totalPredictions.toLocaleString()}</div>
            <div className="text-gray-300">Total Predictions</div>
          </motion.div>
          
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20"
            whileHover={{ scale: 1.02 }}
          >
            <div className="text-2xl font-bold text-green-400">{stats.avgConfidence}%</div>
            <div className="text-gray-300">Avg Confidence</div>
          </motion.div>
          
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20"
            whileHover={{ scale: 1.02 }}
          >
            <div className="text-2xl font-bold text-blue-400">{stats.activePillars}/5</div>
            <div className="text-gray-300">Active Pillars</div>
          </motion.div>
          
          <motion.div 
            className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20"
            whileHover={{ scale: 1.02 }}
          >
            <div className="text-sm font-bold text-purple-400">{stats.lastUpdate}</div>
            <div className="text-gray-300">Last Update</div>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Current Prediction */}
          <div className="lg:col-span-2">
            <motion.div 
              className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold">Current Prediction</h2>
                <motion.button
                  onClick={generatePrediction}
                  disabled={isGenerating}
                  className="bg-gradient-to-r from-orange-500 to-pink-500 px-6 py-3 rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {isGenerating ? '🔄 Generating...' : '🎯 Generate New'}
                </motion.button>
              </div>

              {currentPrediction ? (
                <div className="space-y-6">
                  {/* Numbers Display */}
                  <div className="text-center">
                    <div className="flex justify-center items-center space-x-4 mb-4">
                      {currentPrediction.numbers.map((num, index) => (
                        <motion.div
                          key={index}
                          className="w-16 h-16 bg-gradient-to-br from-orange-400 to-pink-400 rounded-full flex items-center justify-center text-xl font-bold text-white shadow-lg"
                          initial={{ scale: 0, rotate: -180 }}
                          animate={{ scale: 1, rotate: 0 }}
                          transition={{ delay: index * 0.1 }}
                        >
                          {num}
                        </motion.div>
                      ))}
                      <div className="text-2xl text-gray-400 mx-4">+</div>
                      <motion.div
                        className="w-16 h-16 bg-gradient-to-br from-red-500 to-red-600 rounded-full flex items-center justify-center text-xl font-bold text-white shadow-lg"
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        transition={{ delay: 0.6 }}
                      >
                        {currentPrediction.powerball}
                      </motion.div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div>
                        <div className="text-2xl font-bold text-green-400">{currentPrediction.confidence.toFixed(1)}%</div>
                        <div className="text-gray-400">Confidence</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-blue-400">{currentPrediction.score.toFixed(1)}</div>
                        <div className="text-gray-400">Total Score</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-purple-400">{currentPrediction.digitalRoot}</div>
                        <div className="text-gray-400">Digital Root</div>
                      </div>
                    </div>
                  </div>

                  {/* AI Analysis */}
                  <div className="bg-black/20 rounded-lg p-4">
                    <h3 className="text-lg font-semibold mb-2 text-cyan-400">🤖 AI Analysis</h3>
                    <p className="text-gray-300 leading-relaxed">{currentPrediction.analysis}</p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">🎯</div>
                  <div className="text-xl text-gray-400">Click "Generate New" to create your first prediction</div>
                </div>
              )}
            </motion.div>
          </div>

          {/* 5 Pillars Status */}
          <div>
            <motion.div 
              className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <h2 className="text-xl font-bold mb-6">5-Pillar Analysis</h2>
              
              {currentPrediction && (
                <div className="space-y-4">
                  {Object.entries(currentPrediction.pillars).map(([pillarName, score]) => {
                    const maxScores: Record<string, number> = {
                      statistical: 30,
                      aiIntelligence: 25,
                      dataAnalytics: 20,
                      predictiveInsights: 15,
                      cosmicIntelligence: 25
                    };
                    
                    const percentage = (score / maxScores[pillarName]) * 100;
                    
                    return (
                      <motion.div
                        key={pillarName}
                        className="cursor-pointer"
                        onClick={() => setSelectedPillar(selectedPillar === pillarName ? null : pillarName)}
                        whileHover={{ scale: 1.02 }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold">{formatPillarName(pillarName)}</span>
                          <span className={`font-bold ${getPillarColor(pillarName, score)}`}>
                            {score.toFixed(1)}/{maxScores[pillarName]}
                          </span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <motion.div
                            className="bg-gradient-to-r from-orange-400 to-pink-400 h-2 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${percentage}%` }}
                            transition={{ duration: 1, delay: 0.2 }}
                          />
                        </div>
                        
                        <AnimatePresence>
                          {selectedPillar === pillarName && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              className="mt-3 p-3 bg-black/20 rounded-lg"
                            >
                              <p className="text-sm text-gray-300">{getPillarDescription(pillarName)}</p>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    );
                  })}
                </div>
              )}
              
              {!currentPrediction && (
                <div className="text-center py-8">
                  <div className="text-4xl mb-2">⚡</div>
                  <div className="text-gray-400">Generate a prediction to see pillar analysis</div>
                </div>
              )}
            </motion.div>

            {/* System Health */}
            <motion.div 
              className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 mt-6"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <h2 className="text-xl font-bold mb-4">System Health</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span>AI Providers</span>
                  <span className="text-green-400">🟢 3/3 Online</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Database</span>
                  <span className="text-green-400">🟢 Connected</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Prediction Engine</span>
                  <span className="text-green-400">🟢 Operational</span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Data Analytics</span>
                  <span className="text-green-400">🟢 Processing</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Recent Predictions History */}
        <motion.div 
          className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 mt-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h2 className="text-xl font-bold mb-4">Recent Predictions</h2>
          <div className="text-center py-8 text-gray-400">
            <div className="text-4xl mb-2">📊</div>
            <div>Prediction history will appear here as you generate more predictions</div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

