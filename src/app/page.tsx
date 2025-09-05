'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  Brain, 
  TrendingUp, 
  Database, 
  Zap, 
  Target, 
  BarChart3,
  Star,
  Crown,
  Shield,
  CheckCircle,
  ArrowRight,
  X,
  Clock,
  Sparkles,
  Moon,
  Cpu
} from 'lucide-react';

export default function HomePage() {
  const [timeLeft, setTimeLeft] = useState({
    hours: 23,
    minutes: 59,
    seconds: 59
  });
  const [showBanner, setShowBanner] = useState(true);

  // Countdown timer effect
  useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev.seconds > 0) {
          return { ...prev, seconds: prev.seconds - 1 };
        } else if (prev.minutes > 0) {
          return { ...prev, minutes: prev.minutes - 1, seconds: 59 };
        } else if (prev.hours > 0) {
          return { hours: prev.hours - 1, minutes: 59, seconds: 59 };
        }
        return prev;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800">
      {/* Flash Sale Banner */}
      {showBanner && (
        <div className="bg-gradient-to-r from-red-500 to-pink-600 text-white py-3 px-4 relative">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Zap className="w-5 h-5 text-yellow-300" />
              <span className="font-bold">24-HOUR FLASH SALE</span>
              <span className="hidden sm:inline">50% OFF ALL ANALYSIS PACKAGES</span>
              <span className="text-sm">ENDS IN:</span>
              <div className="flex items-center space-x-1 font-mono">
                <span className="bg-black/20 px-2 py-1 rounded">{timeLeft.hours}h</span>
                <span className="bg-black/20 px-2 py-1 rounded">{timeLeft.minutes}m</span>
                <span className="bg-black/20 px-2 py-1 rounded">{timeLeft.seconds}s</span>
              </div>
            </div>
            <button 
              onClick={() => setShowBanner(false)}
              className="text-white/80 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link href="/" className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-orange-400 to-red-500 rounded-full flex items-center justify-center">
                <span className="text-white font-bold text-lg">🔮</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">PatternSight</h1>
                <p className="text-xs text-gray-600">Where Mathematics Meets Possibility</p>
              </div>
            </Link>

            {/* Navigation */}
            <div className="flex items-center space-x-4">
              <Link href="/dashboard" className="text-gray-600 hover:text-gray-900 px-4 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 transition-all">
                Dashboard
              </Link>
              <button className="text-gray-600 hover:text-gray-900 px-4 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 transition-all">
                Sign In
              </button>
              <button className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-400 hover:to-red-400 text-white px-6 py-2 rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl">
                Get Started
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 text-white">
        <div className="max-w-7xl mx-auto px-4 py-12">
          {/* Launch Badge */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center space-x-2 bg-green-500 text-white px-6 py-2 rounded-full font-semibold">
              <CheckCircle className="w-5 h-5" />
              <span>NOW LIVE - PUBLIC LAUNCH</span>
            </div>
          </div>

          {/* Hero Section */}
          <div className="text-center mb-12">
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-6xl md:text-7xl font-bold mb-6"
            >
              <span className="bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
                PatternSight v4.0
              </span>
              <br />
              <span className="text-white">
                Ultimate AI Prediction Platform
              </span>
            </motion.h1>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="text-xl text-white/90 mb-4 max-w-4xl mx-auto"
            >
              The world's most advanced lottery prediction system combining
              <br />
              <strong>10 mathematical pillars + 3 AI enhancement add-ons</strong>
            </motion.p>
            
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-2xl font-bold text-orange-400 mb-8"
            >
              🎯 18-20% Pattern Accuracy • 🧠 Multi-AI Intelligence • 🌙 Cosmic Enhancement
            </motion.p>

            {/* System Status Indicators */}
            <div className="flex items-center justify-center space-x-8 mb-12">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-white/90">🏛️ 10 Pillars Active</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse"></div>
                <span className="text-white/90">🧠 Multi-AI Online</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
                <span className="text-white/90">🌙 Cosmic Intelligence</span>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex items-center justify-center space-x-4 mb-16">
              <Link href="/dashboard">
                <button className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-400 hover:to-red-400 text-white px-8 py-4 rounded-lg font-bold text-lg shadow-lg hover:shadow-xl transition-all transform hover:scale-105 flex items-center space-x-2">
                  <span>🔮 Discover Your Patterns</span>
                </button>
              </Link>
              <button className="bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-white px-8 py-4 rounded-lg font-bold text-lg shadow-lg hover:shadow-xl transition-all transform hover:scale-105 flex items-center space-x-2">
                <span>See How It Works</span>
                <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>

        {/* Stats Section - Orange Background like original */}
        <div className="bg-gradient-to-r from-orange-500 to-red-500 py-12">
          <div className="max-w-7xl mx-auto px-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center text-white">
              <div>
                <div className="text-4xl font-bold mb-2">18-20%</div>
                <div className="text-lg">Pattern Accuracy</div>
                <div className="text-sm opacity-80">vs 0.007% random</div>
              </div>
              <div>
                <div className="text-4xl font-bold mb-2">10+3</div>
                <div className="text-lg">AI Pillars + Add-ons</div>
                <div className="text-sm opacity-80">Mathematical + AI</div>
              </div>
              <div>
                <div className="text-4xl font-bold mb-2">5+ Years</div>
                <div className="text-lg">Historical Data</div>
                <div className="text-sm opacity-80">Real lottery analysis</div>
              </div>
              <div>
                <div className="text-4xl font-bold mb-2">3 AI</div>
                <div className="text-lg">Enhancement Add-ons</div>
                <div className="text-sm opacity-80">$5.99 each</div>
              </div>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="py-16">
          <div className="max-w-7xl mx-auto px-4">
            <h2 className="text-4xl font-bold text-white text-center mb-12">Revolutionary 10-Pillar Architecture</h2>
            <p className="text-xl text-white/90 text-center mb-12 max-w-4xl mx-auto">
              Our sophisticated system combines 10 peer-reviewed mathematical pillars with 3 optional AI enhancement add-ons for unprecedented prediction capabilities.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">Mathematical Foundation</h3>
                <p className="text-white/80">10 peer-reviewed pillars including CDM Bayesian, Order Statistics, Ensemble Deep Learning, and Markov Chain analysis.</p>
              </div>

              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Brain className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">Multi-AI Intelligence</h3>
                <p className="text-white/80">Advanced AI reasoning with OpenAI GPT-4, Claude, and DeepSeek integration for contextual pattern analysis.</p>
              </div>

              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Database className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">Real Data Analysis</h3>
                <p className="text-white/80">5+ years of historical lottery data across multiple systems with proven statistical significance (P-value &lt; 0.01).</p>
              </div>

              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <TrendingUp className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-white mb-3">Honest Performance</h3>
                <p className="text-white/80">Transparent 18-20% pattern accuracy with complete mathematical validation and realistic expectations.</p>
              </div>
            </div>

            {/* Add-On Showcase */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
              <h3 className="text-3xl font-bold text-white text-center mb-8">Premium AI Enhancement Add-Ons</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                <div className="text-center">
                  <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Moon className="w-10 h-10 text-white" />
                  </div>
                  <h4 className="text-xl font-bold text-white mb-2">🌙 Cosmic Intelligence</h4>
                  <p className="text-white/80 mb-3">Lunar phases, zodiac alignments, numerological patterns, and sacred geometry analysis.</p>
                  <div className="text-orange-400 font-bold">$5.99/month</div>
                </div>

                <div className="text-center">
                  <div className="w-20 h-20 bg-gradient-to-r from-cyan-500 to-teal-500 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Cpu className="w-10 h-10 text-white" />
                  </div>
                  <h4 className="text-xl font-bold text-white mb-2">🧠 Claude Nexus Intelligence</h4>
                  <p className="text-white/80 mb-3">5-engine AI system with statistical, neural network, quantum, and pattern recognition engines.</p>
                  <div className="text-orange-400 font-bold">$5.99/month</div>
                </div>

                <div className="text-center">
                  <div className="w-20 h-20 bg-gradient-to-r from-pink-500 to-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Sparkles className="w-10 h-10 text-white" />
                  </div>
                  <h4 className="text-xl font-bold text-white mb-2">💎 Premium Enhancement</h4>
                  <p className="text-white/80 mb-3">Ultimate multi-model AI ensemble with predictive intelligence and market analysis.</p>
                  <div className="text-orange-400 font-bold">$5.99/month</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Pricing Section */}
        <div className="py-16">
          <div className="max-w-7xl mx-auto px-4">
            <h2 className="text-4xl font-bold text-white text-center mb-4">Choose Your Intelligence Level</h2>
            <p className="text-xl text-white/90 text-center mb-12">Flexible subscription tiers with optional AI enhancement add-ons</p>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
              {/* Lite Plan */}
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20">
                <h3 className="text-2xl font-bold text-white mb-2">Pattern Lite</h3>
                <div className="text-4xl font-bold text-white mb-4">FREE</div>
                <p className="text-white/80 mb-6">3 analyses per day</p>
                <ul className="space-y-3 mb-8">
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Basic pattern analysis</span>
                  </li>
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ 10 mathematical pillars</span>
                  </li>
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Community access</span>
                  </li>
                </ul>
                <button className="w-full bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-400 hover:to-cyan-400 text-white py-3 rounded-lg font-semibold transition-all">
                  Get Started
                </button>
              </div>

              {/* Starter Plan */}
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20">
                <h3 className="text-2xl font-bold text-white mb-2">Pattern Starter</h3>
                <div className="text-4xl font-bold text-white mb-4">$9.99</div>
                <p className="text-white/80 mb-6">10 analyses per day</p>
                <ul className="space-y-3 mb-8">
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Enhanced pattern analysis</span>
                  </li>
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Daily insights</span>
                  </li>
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Can purchase add-ons</span>
                  </li>
                </ul>
                <button className="w-full bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-400 hover:to-emerald-400 text-white py-3 rounded-lg font-semibold transition-all">
                  Upgrade
                </button>
              </div>

              {/* Pro Plan */}
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border-2 border-orange-500 relative">
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <span className="bg-orange-500 text-white px-4 py-1 rounded-full text-sm font-semibold">Most Popular</span>
                </div>
                <h3 className="text-2xl font-bold text-white mb-2">Pattern Pro</h3>
                <div className="text-4xl font-bold text-white mb-4">$39.99</div>
                <p className="text-white/80 mb-6">50 analyses per day</p>
                <ul className="space-y-3 mb-8">
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Advanced AI analysis</span>
                  </li>
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Choose 2 add-ons included</span>
                  </li>
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Priority support</span>
                  </li>
                </ul>
                <button className="w-full bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-400 hover:to-red-400 text-white py-3 rounded-lg font-semibold transition-all">
                  Upgrade Now
                </button>
              </div>

              {/* Elite Plan */}
              <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20">
                <h3 className="text-2xl font-bold text-white mb-2">Pattern Elite</h3>
                <div className="text-4xl font-bold text-white mb-4">$199.99</div>
                <p className="text-white/80 mb-6">300 analyses per day</p>
                <ul className="space-y-3 mb-8">
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ Maximum AI power</span>
                  </li>
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ All 3 add-ons included</span>
                  </li>
                  <li className="flex items-center space-x-2 text-white/90">
                    <CheckCircle className="w-5 h-5 text-green-400" />
                    <span>✓ VIP support</span>
                  </li>
                </ul>
                <button className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-400 hover:to-pink-400 text-white py-3 rounded-lg font-semibold transition-all">
                  Go Elite
                </button>
              </div>
            </div>

            {/* Add-On Pricing */}
            <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
              <h3 className="text-2xl font-bold text-white text-center mb-6">AI Enhancement Add-Ons</h3>
              <p className="text-white/80 text-center mb-8">Enhance any subscription with powerful AI add-ons - $5.99/month each</p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white/5 rounded-lg p-4 text-center">
                  <div className="text-2xl mb-2">🌙</div>
                  <div className="font-bold text-white">Cosmic Intelligence</div>
                  <div className="text-orange-400 font-bold">$5.99/month</div>
                </div>
                <div className="bg-white/5 rounded-lg p-4 text-center">
                  <div className="text-2xl mb-2">🧠</div>
                  <div className="font-bold text-white">Claude Nexus Intelligence</div>
                  <div className="text-orange-400 font-bold">$5.99/month</div>
                </div>
                <div className="bg-white/5 rounded-lg p-4 text-center">
                  <div className="text-2xl mb-2">💎</div>
                  <div className="font-bold text-white">Premium Enhancement</div>
                  <div className="text-orange-400 font-bold">$5.99/month</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Flash Sale Section */}
        <div className="py-16">
          <div className="max-w-7xl mx-auto px-4">
            <div className="bg-gradient-to-r from-red-600 to-purple-600 rounded-xl p-8">
              <h3 className="text-3xl font-bold text-white text-center mb-2">⚡ FLASH SALE: Analysis Bundles</h3>
              <p className="text-xl text-white/90 text-center mb-8">50% OFF - Never expire, use anytime!</p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center">
                  <h4 className="text-xl font-bold text-white mb-2">25 Analyses</h4>
                  <div className="text-2xl font-bold text-white mb-1">
                    <span className="line-through text-white/60">$12.99</span> $6.49
                  </div>
                  <p className="text-white/80">$0.26 each</p>
                </div>
                
                <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center">
                  <h4 className="text-xl font-bold text-white mb-2">50 Analyses</h4>
                  <div className="text-2xl font-bold text-white mb-1">
                    <span className="line-through text-white/60">$22.99</span> $11.49
                  </div>
                  <p className="text-white/80">$0.23 each</p>
                </div>
                
                <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center border-2 border-yellow-400 relative">
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="bg-yellow-400 text-black px-3 py-1 rounded-full text-sm font-bold">BEST VALUE</span>
                  </div>
                  <h4 className="text-xl font-bold text-white mb-2">125 Analyses</h4>
                  <div className="text-2xl font-bold text-white mb-1">
                    <span className="line-through text-white/60">$49.99</span> $24.99
                  </div>
                  <p className="text-white/80">$0.20 each</p>
                </div>
                
                <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-center">
                  <h4 className="text-xl font-bold text-white mb-2">250 Analyses</h4>
                  <div className="text-2xl font-bold text-white mb-1">
                    <span className="line-through text-white/60">$89.99</span> $44.99
                  </div>
                  <p className="text-white/80">$0.18 each</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <Footer />
    </div>
  );
}
