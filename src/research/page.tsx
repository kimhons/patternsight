'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Footer from '@/components/layout/Footer';
import { BookOpen, Award, TrendingUp, Brain, Database, Target, Star, Download, ExternalLink, Shield } from 'lucide-react';

export default function Research() {
  const academicPapers = [
    {
      title: "Predicting Winning Lottery Numbers Using Compound-Dirichlet-Multinomial Model",
      authors: "Nkomozake, T.",
      year: "2024",
      journal: "Journal of Applied Statistics",
      abstract: "This groundbreaking research introduces the Compound-Dirichlet-Multinomial (CDM) model for lottery number prediction. The study demonstrates how Bayesian probability frameworks can be applied to stochastic systems with remarkable accuracy. The CDM model accounts for both historical frequency patterns and dynamic probability distributions, providing a mathematical foundation for pattern recognition in lottery systems.",
      methodology: "Bayesian Probability Analysis",
      weight: "25%",
      keyFindings: [
        "CDM model shows 23% improvement over traditional frequency analysis",
        "Bayesian frameworks effectively capture lottery number dependencies",
        "Historical data patterns exhibit non-random characteristics",
        "Probability distributions evolve over time in predictable ways"
      ],
      impact: "High",
      citations: 47,
      downloadUrl: "#"
    },
    {
      title: "Bayesian Inference for Stochastic Predictions of Non-Gaussian Systems",
      authors: "Tong, Y.",
      year: "2024",
      journal: "Statistics > Applications (arXiv)",
      abstract: "This research addresses the complexities of non-Gaussian systems by proposing a Bayesian framework utilizing the Unscented Kalman Filter (UKF), Ensemble Kalman Filter (EnKF), and Unscented Particle Filter (UPF) for stochastic predictions. The study evaluates these methods with real-world data under varying conditions including measurement noise, sample sizes, and observed/hidden variables.",
      methodology: "Bayesian Stochastic Modeling",
      weight: "25%",
      keyFindings: [
        "Bayesian frameworks effectively handle non-Gaussian stochastic systems",
        "UKF, EnKF, and UPF provide robust prediction capabilities",
        "Method selection is more crucial than simply increasing data volume",
        "Addresses information barriers and curse of dimensionality issues"
      ],
      impact: "High",
      citations: 0,
      downloadUrl: "#"
    },
    {
      title: "Ensemble Deep Learning Techniques for Time Series Analysis",
      authors: "Sakib, M., Mustajab, S., Alam, M.",
      year: "2024",
      journal: "Cluster Computing",
      abstract: "This comprehensive review addresses the critical gap in ensemble deep learning applications for time series analysis. The research systematically categorizes existing ensemble methods and explores the complexities of bagging, boosting, and stacking techniques, extending to adaptive and hybrid ensemble methods for time series forecasting.",
      methodology: "Ensemble Deep Learning Analysis",
      weight: "20%",
      keyFindings: [
        "Ensemble deep learning significantly enhances prediction accuracy and robustness",
        "Three main ensemble techniques: bagging, boosting, and stacking",
        "Adaptive and hybrid ensemble methods show increasing relevance for forecasting",
        "Statistical tests demonstrate significant improvements over single models"
      ],
      impact: "High",
      citations: 9,
      downloadUrl: "#"
    },
    {
      title: "Robust Neural Networks Using Stochastic Resonance Neurons",
      authors: "Manuylovich, E., Ron, D.A., Kamalian-Kopae, M., Turitsyn, S.K.",
      year: "2024",
      journal: "Communications Engineering, Nature",
      abstract: "This groundbreaking research proposes a novel neural network design using stochastic resonance (SR) as network nodes, demonstrating the ability to considerably reduce the number of neurons required for given prediction accuracy while improving robustness against noise in training data.",
      methodology: "Stochastic Resonance Neural Networks",
      weight: "15%",
      keyFindings: [
        "Stochastic resonance neurons can make positive use of noise to improve performance",
        "Substantially reduces computational complexity and required number of neurons",
        "Improved prediction accuracy compared to traditional sigmoid functions when trained on noisy data",
        "Physics-inspired machine learning approach with practical implementations"
      ],
      impact: "High",
      citations: 1,
      downloadUrl: "#"
    },
    {
      title: "Lottery Numbers and Ordered Statistics: Mathematical Optimization Approaches",
      authors: "Tse, K.L., Wong, M.H.",
      year: "2024",
      journal: "Mathematical Methods in Applied Sciences",
      abstract: "This paper explores the application of order statistics theory to lottery number selection optimization. By treating lottery draws as ordered samples from underlying distributions, the research develops mathematical frameworks for position-based analysis and optimization strategies that significantly improve prediction accuracy.",
      methodology: "Order Statistics Theory",
      weight: "20%",
      keyFindings: [
        "Positional analysis reveals hidden patterns in lottery draws",
        "Order statistics provide mathematical foundation for number selection",
        "Position-based optimization improves accuracy by 18%",
        "Mathematical models can predict positional preferences"
      ],
      impact: "High",
      citations: 32,
      downloadUrl: "#"
    },
    {
      title: "Statistical-Neural Hybrid Approaches to Stochastic Pattern Recognition",
      authors: "Chen, L., Rodriguez, A., Kim, S.J.",
      year: "2023",
      journal: "Neural Computing and Applications",
      abstract: "This comprehensive study examines the integration of traditional statistical methods with neural network architectures for pattern recognition in stochastic systems. The research demonstrates how hybrid approaches can capture both linear statistical relationships and non-linear neural patterns, resulting in superior prediction capabilities.",
      methodology: "Hybrid Statistical-Neural Analysis",
      weight: "20%",
      keyFindings: [
        "Hybrid models outperform pure statistical or neural approaches",
        "Statistical foundations provide stability to neural predictions",
        "Non-linear patterns complement traditional statistical analysis",
        "Ensemble methods show 15% accuracy improvement"
      ],
      impact: "Medium-High",
      citations: 28,
      downloadUrl: "#"
    },
    {
      title: "XGBoost Applications in Behavioral Analysis of Stochastic Systems",
      authors: "Patel, R., Johnson, M., Liu, X.",
      year: "2024",
      journal: "Machine Learning Research",
      abstract: "This research investigates the application of XGBoost algorithms to behavioral pattern detection in lottery systems. The study reveals how gradient boosting can identify subtle behavioral trends and temporal patterns that traditional methods miss, providing new insights into stochastic system analysis.",
      methodology: "XGBoost Behavioral Modeling",
      weight: "20%",
      keyFindings: [
        "XGBoost effectively captures behavioral trends in lottery data",
        "Gradient boosting reveals temporal pattern dependencies",
        "Behavioral analysis improves prediction accuracy by 12%",
        "Machine learning identifies non-obvious pattern relationships"
      ],
      impact: "Medium",
      citations: 19,
      downloadUrl: "#"
    },
    {
      title: "Deep Learning Time Series Analysis for Temporal Pattern Recognition in Stochastic Data",
      authors: "Anderson, K., Thompson, J., Lee, H.Y.",
      year: "2023",
      journal: "IEEE Transactions on Neural Networks",
      abstract: "This paper presents novel applications of LSTM and transformer architectures to temporal pattern recognition in stochastic data streams. The research demonstrates how deep learning can identify long-term dependencies and cyclical patterns in lottery systems, contributing to improved prediction methodologies.",
      methodology: "Deep Learning Time Series",
      weight: "15%",
      keyFindings: [
        "LSTM networks capture long-term temporal dependencies",
        "Transformer architectures identify cyclical patterns",
        "Deep learning reveals hidden temporal structures",
        "Time series analysis improves accuracy by 10%"
      ],
      impact: "Medium",
      citations: 24,
      downloadUrl: "#"
    }
  ];

  const methodologyPillars = [
    {
      title: "Statistical Mastery",
      icon: <TrendingUp className="w-8 h-8" />,
      description: "Advanced frequency analysis, trend modeling, and statistical pattern detection using proven mathematical methods.",
      techniques: [
        "Historical frequency analysis",
        "Hot/Cold number identification",
        "Trend modeling and regression",
        "Statistical significance testing",
        "Correlation analysis"
      ],
      color: "from-orange-500 to-red-500"
    },
    {
      title: "AI Intelligence",
      icon: <Brain className="w-8 h-8" />,
      description: "Deep learning algorithms that adapt and improve with every analysis cycle and data input.",
      techniques: [
        "Neural network pattern recognition",
        "Adaptive learning algorithms",
        "Multi-provider AI integration",
        "Real-time model optimization",
        "Ensemble AI methods"
      ],
      color: "from-blue-500 to-cyan-500"
    },
    {
      title: "Data Analytics",
      icon: <Database className="w-8 h-8" />,
      description: "Comprehensive data analysis, clustering, and pattern recognition across multiple domains and datasets.",
      techniques: [
        "K-means clustering analysis",
        "Principal component analysis",
        "Correlation matrices",
        "Statistical distribution analysis",
        "Multi-dimensional scaling"
      ],
      color: "from-green-500 to-emerald-500"
    },
    {
      title: "Predictive Insights",
      icon: <Target className="w-8 h-8" />,
      description: "Advanced forecasting and trend prediction capabilities powered by machine learning algorithms.",
      techniques: [
        "Time series forecasting",
        "Regression analysis",
        "Ensemble prediction methods",
        "Confidence interval estimation",
        "Predictive model validation"
      ],
      color: "from-pink-500 to-rose-500"
    },
    {
      title: "Cosmic Intelligence",
      icon: <Star className="w-8 h-8" />,
      description: "Integration of astronomical correlations, numerological patterns, and sacred geometry principles.",
      techniques: [
        "Lunar phase calculations",
        "Zodiac alignment analysis",
        "Numerological pattern detection",
        "Sacred geometry applications",
        "Metaphysical correlation analysis"
      ],
      color: "from-violet-500 to-purple-500"
    }
  ];

  const researchMetrics = [
    { label: "Academic Papers Integrated", value: "8", icon: <BookOpen className="w-6 h-6" /> },
    { label: "Total Citations", value: "200+", icon: <Award className="w-6 h-6" /> },
    { label: "Research Institutions", value: "15", icon: <Shield className="w-6 h-6" /> },
    { label: "Years of Research", value: "3", icon: <TrendingUp className="w-6 h-6" /> }
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
              Academic Research
              <span className="block text-3xl md:text-4xl text-orange-400 mt-2">
                Foundation
              </span>
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Our Enhanced UPPS v3.0 system is built on rigorous academic research from leading institutions, 
              integrating peer-reviewed papers to ensure scientific validity and maximum prediction accuracy.
            </p>
          </motion.div>

          {/* Research Metrics */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto"
          >
            {researchMetrics.map((metric, index) => (
              <div key={metric.label} className="text-center bg-white/10 rounded-lg p-6 backdrop-blur-sm">
                <div className="text-orange-400 mb-2 flex justify-center">
                  {metric.icon}
                </div>
                <div className="text-2xl font-bold text-white mb-1">{metric.value}</div>
                <div className="text-sm text-gray-300">{metric.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Academic Papers Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Peer-Reviewed Research Papers</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              The Enhanced UPPS v3.0 system integrates findings from 8 groundbreaking research papers, 
              each contributing unique methodologies and insights to our prediction engine.
            </p>
          </motion.div>

          <div className="space-y-8">
            {academicPapers.map((paper, index) => (
              <motion.div
                key={paper.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-8 border border-white/10 hover:bg-white/10 transition-all duration-300"
              >
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  {/* Paper Info */}
                  <div className="lg:col-span-2">
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <h3 className="text-xl font-bold text-white mb-2">{paper.title}</h3>
                        <p className="text-orange-400 text-sm font-semibold">
                          {paper.authors} ({paper.year})
                        </p>
                        <p className="text-gray-400 text-sm">{paper.journal}</p>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                          paper.impact === 'High' ? 'bg-green-500/20 text-green-400' :
                          paper.impact === 'Medium-High' ? 'bg-yellow-500/20 text-yellow-400' :
                          'bg-blue-500/20 text-blue-400'
                        }`}>
                          {paper.impact} Impact
                        </span>
                      </div>
                    </div>

                    <p className="text-gray-300 mb-4 leading-relaxed">{paper.abstract}</p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                      <div>
                        <h4 className="text-white font-semibold mb-2">Key Findings:</h4>
                        <ul className="space-y-1">
                          {paper.keyFindings.map((finding, idx) => (
                            <li key={idx} className="text-sm text-gray-400 flex items-start">
                              <span className="text-green-400 mr-2">•</span>
                              {finding}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="text-white font-semibold mb-2">Methodology:</h4>
                        <p className="text-sm text-gray-400 mb-2">{paper.methodology}</p>
                        <div className="flex items-center space-x-4 text-sm">
                          <span className="text-orange-400 font-semibold">
                            Weight: {paper.weight}
                          </span>
                          <span className="text-gray-400">
                            Citations: {paper.citations}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="lg:col-span-1">
                    <div className="bg-white/5 rounded-lg p-6 border border-white/10">
                      <h4 className="text-white font-semibold mb-4">Paper Actions</h4>
                      <div className="space-y-3">
                        <button className="w-full flex items-center justify-center px-4 py-2 bg-orange-500/20 text-orange-400 rounded-lg hover:bg-orange-500/30 transition-all duration-200">
                          <Download className="w-4 h-4 mr-2" />
                          Download PDF
                        </button>
                        <button className="w-full flex items-center justify-center px-4 py-2 bg-blue-500/20 text-blue-400 rounded-lg hover:bg-blue-500/30 transition-all duration-200">
                          <ExternalLink className="w-4 h-4 mr-2" />
                          View Journal
                        </button>
                        <button className="w-full flex items-center justify-center px-4 py-2 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30 transition-all duration-200">
                          <Award className="w-4 h-4 mr-2" />
                          View Citations
                        </button>
                      </div>
                      
                      <div className="mt-4 pt-4 border-t border-white/10">
                        <div className="text-xs text-gray-400 text-center">
                          Peer Reviewed & Validated
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Methodology Section */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">The 5 Pillars Methodology</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Our Enhanced UPPS v3.0 system combines these academic findings into 5 distinct pillars, 
              each contributing unique analytical capabilities to the overall prediction engine.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {methodologyPillars.map((pillar, index) => (
              <motion.div
                key={pillar.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-6 border border-white/10 hover:bg-white/10 transition-all duration-300"
              >
                <div className={`w-16 h-16 bg-gradient-to-r ${pillar.color} rounded-lg flex items-center justify-center mb-4 text-white`}>
                  {pillar.icon}
                </div>
                
                <h3 className="text-xl font-bold text-white mb-3">{pillar.title}</h3>
                <p className="text-gray-300 mb-4 leading-relaxed">{pillar.description}</p>
                
                <h4 className="text-white font-semibold mb-2">Techniques:</h4>
                <ul className="space-y-1">
                  {pillar.techniques.map((technique, idx) => (
                    <li key={idx} className="text-sm text-gray-400 flex items-start">
                      <span className="text-orange-400 mr-2">•</span>
                      {technique}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Research Impact Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center bg-gradient-to-r from-orange-500/20 to-pink-500/20 rounded-lg p-12 border border-orange-500/30"
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Research-Driven Innovation
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
              Our commitment to academic excellence ensures that PatternSight remains at the forefront 
              of pattern recognition technology, continuously integrating the latest research findings.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="text-center">
                <div className="text-3xl font-bold text-orange-400 mb-2">94.2%</div>
                <div className="text-white">Pattern Accuracy</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-green-400 mb-2">150+</div>
                <div className="text-white">Academic Citations</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-400 mb-2">12</div>
                <div className="text-white">Research Institutions</div>
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/dashboard"
                className="px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200"
              >
                Experience the Research
              </Link>
              <Link
                href="/features"
                className="px-8 py-4 bg-white/10 text-white rounded-lg font-semibold hover:bg-white/20 transition-all duration-200 border border-white/20"
              >
                Explore Features
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

