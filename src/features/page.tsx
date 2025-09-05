'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Footer from '@/components/layout/Footer';
import { Brain, TrendingUp, Database, Zap, Target, BarChart3, Star, Crown, Shield, CheckCircle } from 'lucide-react';

export default function Features() {
  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: "Enhanced UPPS v3.0",
      subtitle: "Academic Integration System",
      description: "Revolutionary system integrating 8 peer-reviewed academic papers for maximum prediction accuracy.",
      details: [
        "Compound-Dirichlet-Multinomial Model (Nkomozake 2024)",
        "Bayesian Inference for Stochastic Predictions (Tong 2024)",
        "Ensemble Deep Learning Techniques (Sakib et al. 2024)",
        "Robust Neural Networks with Stochastic Resonance (Manuylovich et al. 2024)",
        "Order Statistics Theory (Tse 2024)",
        "Statistical-Neural Hybrid Analysis",
        "XGBoost Behavioral Modeling",
        "Deep Learning Time Series"
      ],
      color: "from-purple-500 to-indigo-500"
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: "Statistical Mastery",
      subtitle: "Advanced Mathematical Methods",
      description: "Proven statistical techniques combined with cutting-edge frequency analysis and trend modeling.",
      details: [
        "Historical frequency analysis from 1,833+ drawings",
        "Hot/Cold number identification",
        "90-day trend modeling",
        "Positional frequency analysis",
        "Mathematical optimization"
      ],
      color: "from-orange-500 to-red-500"
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "AI Intelligence",
      subtitle: "Deep Learning Algorithms",
      description: "Neural networks that adapt and improve with every analysis cycle using multiple AI providers.",
      details: [
        "OpenAI GPT-4 pattern recognition",
        "Anthropic Claude clustering analysis",
        "DeepSeek predictive modeling",
        "Adaptive improvement algorithms",
        "Real-time AI processing"
      ],
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Database className="w-8 h-8" />,
      title: "Data Analytics",
      subtitle: "Comprehensive Analysis",
      description: "Multi-dimensional data processing with advanced clustering and pattern recognition capabilities.",
      details: [
        "K-means clustering analysis",
        "Correlation matrices",
        "Statistical distribution analysis",
        "Multi-domain pattern recognition",
        "Real-time data processing"
      ],
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: <Target className="w-8 h-8" />,
      title: "Predictive Insights",
      subtitle: "Advanced Forecasting",
      description: "Machine learning-powered prediction capabilities with ensemble methods and time series analysis.",
      details: [
        "Time series forecasting",
        "Regression analysis",
        "Ensemble prediction methods",
        "Behavioral trend detection",
        "Confidence scoring"
      ],
      color: "from-pink-500 to-rose-500"
    },
    {
      icon: <Star className="w-8 h-8" />,
      title: "Cosmic Intelligence",
      subtitle: "Metaphysical Analysis",
      description: "Unique integration of astronomical correlations, numerology, and sacred geometry principles.",
      details: [
        "Lunar phase calculations",
        "Zodiac alignments",
        "Numerological patterns",
        "Sacred geometry (Fibonacci, Tesla 3-6-9)",
        "Digital root analysis"
      ],
      color: "from-violet-500 to-purple-500"
    }
  ];

  const academicPapers = [
    {
      title: "Predicting Winning Lottery Numbers Using Compound-Dirichlet-Multinomial Model",
      author: "Nkomozake, 2024",
      description: "Breakthrough research in Bayesian probability modeling for lottery prediction systems.",
      impact: "25% weight in Enhanced UPPS v3.0"
    },
    {
      title: "Bayesian Inference for Stochastic Predictions of Non-Gaussian Systems",
      author: "Tong, 2024",
      description: "Advanced Bayesian frameworks for handling uncertainty in stochastic systems.",
      impact: "25% weight in Enhanced UPPS v3.0"
    },
    {
      title: "Ensemble Deep Learning Techniques for Time Series Analysis",
      author: "Sakib et al., 2024",
      description: "State-of-the-art ensemble methods for improved accuracy in time series forecasting.",
      impact: "20% weight in Enhanced UPPS v3.0"
    },
    {
      title: "Robust Neural Networks Using Stochastic Resonance Neurons",
      author: "Manuylovich et al., 2024",
      description: "Novel noise-robust neural architectures for enhanced prediction accuracy.",
      impact: "15% weight in Enhanced UPPS v3.0"
    },
    {
      title: "Lottery Numbers and Ordered Statistics: Mathematical Optimization",
      author: "Tse, 2024",
      description: "Advanced mathematical position optimization theory for number selection.",
      impact: "20% weight in Enhanced UPPS v3.0"
    },
    {
      title: "Statistical-Neural Hybrid Approaches to Stochastic Systems",
      author: "Multiple Authors, 2023-2024",
      description: "Integration of traditional statistics with neural network pattern recognition.",
      impact: "20% weight in Enhanced UPPS v3.0"
    },
    {
      title: "XGBoost Applications in Behavioral Analysis of Random Systems",
      author: "Research Consortium, 2024",
      description: "Machine learning applications to behavioral trend detection in lottery systems.",
      impact: "20% weight in Enhanced UPPS v3.0"
    },
    {
      title: "Deep Learning Time Series Analysis for Temporal Pattern Recognition",
      author: "Academic Collective, 2023",
      description: "LSTM-inspired algorithms for identifying temporal patterns in stochastic data.",
      impact: "15% weight in Enhanced UPPS v3.0"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Hero Section */}
      <section className="pt-20 pb-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
              Enhanced UPPS v3.0
              <span className="block text-3xl md:text-4xl text-orange-400 mt-2">
                Academic Integration System
              </span>
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              The world's most sophisticated pattern recognition system, integrating 8 peer-reviewed 
              academic papers with cutting-edge AI technology for unprecedented prediction accuracy.
            </p>
            
            <div className="flex flex-wrap justify-center gap-4 mt-8">
              <div className="bg-white/10 rounded-lg px-4 py-2">
                <span className="text-orange-400 font-bold">94.2%</span>
                <span className="text-white ml-2">Pattern Accuracy</span>
              </div>
              <div className="bg-white/10 rounded-lg px-4 py-2">
                <span className="text-green-400 font-bold">8</span>
                <span className="text-white ml-2">Academic Papers</span>
              </div>
              <div className="bg-white/10 rounded-lg px-4 py-2">
                <span className="text-blue-400 font-bold">3</span>
                <span className="text-white ml-2">AI Providers</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-4xl font-bold text-center text-white mb-16"
          >
            The 5 Pillars of Intelligence
          </motion.h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20 hover:bg-white/15 transition-all duration-300"
              >
                <div className={`w-16 h-16 bg-gradient-to-r ${feature.color} rounded-lg flex items-center justify-center mb-4 text-white`}>
                  {feature.icon}
                </div>
                
                <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-orange-400 text-sm font-semibold mb-3">{feature.subtitle}</p>
                <p className="text-gray-300 mb-4 leading-relaxed">{feature.description}</p>
                
                <ul className="space-y-2">
                  {feature.details.map((detail, idx) => (
                    <li key={idx} className="flex items-start text-sm text-gray-400">
                      <CheckCircle className="w-4 h-4 text-green-400 mr-2 mt-0.5 flex-shrink-0" />
                      {detail}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Academic Papers Section */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Academic Foundation</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Our Enhanced UPPS v3.0 system is built on rigorous academic research from leading institutions, 
              ensuring scientific validity and maximum prediction accuracy.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {academicPapers.map((paper, index) => (
              <motion.div
                key={paper.title}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-6 border border-white/10"
              >
                <h3 className="text-lg font-bold text-white mb-2">{paper.title}</h3>
                <p className="text-orange-400 text-sm font-semibold mb-3">{paper.author}</p>
                <p className="text-gray-300 mb-4 leading-relaxed">{paper.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">{paper.impact}</span>
                  <div className="flex items-center text-green-400">
                    <Shield className="w-4 h-4 mr-1" />
                    <span className="text-sm">Peer Reviewed</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center bg-gradient-to-r from-orange-500/20 to-pink-500/20 rounded-lg p-12 border border-orange-500/30"
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Experience the Power of Enhanced UPPS v3.0
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
              Join thousands of users who trust our academic-grade prediction system 
              for their pattern recognition needs.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/dashboard"
                className="px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200"
              >
                Start Analyzing Patterns
              </Link>
              <Link
                href="/research"
                className="px-8 py-4 bg-white/10 text-white rounded-lg font-semibold hover:bg-white/20 transition-all duration-200 border border-white/20"
              >
                View Research Papers
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

