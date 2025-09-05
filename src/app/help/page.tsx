'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Footer from '@/components/layout/Footer';
import { Search, Rocket, Target, User, Settings, Mail, MessageCircle, Shield, ChevronRight, HelpCircle, BookOpen, Star, Clock } from 'lucide-react';

export default function HelpCenter() {
  const [searchQuery, setSearchQuery] = useState('');

  const helpTopics = [
    {
      icon: <Rocket className="w-8 h-8" />,
      title: "Getting Started",
      description: "Learn the basics of PatternSight",
      color: "from-blue-500 to-cyan-500",
      articles: [
        "Creating Your Account",
        "Understanding Subscription Plans", 
        "Your First Prediction",
        "Dashboard Overview"
      ]
    },
    {
      icon: <Target className="w-8 h-8" />,
      title: "Using Predictions",
      description: "Master our prediction system",
      color: "from-purple-500 to-pink-500",
      articles: [
        "How Predictions Work",
        "Understanding Confidence Scores",
        "Methodology Breakdown",
        "Interpreting Results"
      ]
    },
    {
      icon: <User className="w-8 h-8" />,
      title: "Account Management",
      description: "Manage your PatternSight account",
      color: "from-green-500 to-emerald-500",
      articles: [
        "Subscription Management",
        "Billing and Payments",
        "Account Settings",
        "Profile Updates"
      ]
    },
    {
      icon: <Settings className="w-8 h-8" />,
      title: "Troubleshooting",
      description: "Resolve common issues",
      color: "from-orange-500 to-red-500",
      articles: [
        "Login Issues",
        "Payment Problems",
        "Technical Support",
        "Feature Requests"
      ]
    }
  ];

  const quickLinks = [
    {
      icon: <Mail className="w-6 h-6" />,
      title: "Contact Support",
      description: "Get help from our team",
      href: "/contact"
    },
    {
      icon: <MessageCircle className="w-6 h-6" />,
      title: "Send Feedback",
      description: "Share your thoughts",
      href: "/contact"
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "Responsible Gaming",
      description: "Play safely and responsibly",
      href: "/responsible-gaming"
    }
  ];

  const popularArticles = [
    {
      title: "How accurate are predictions?",
      description: "Understanding the entertainment nature of our predictions",
      readTime: "3 min read",
      rating: 4.8
    },
    {
      title: "Subscription plan differences",
      description: "Comparing Starter, Plus, and Pro features",
      readTime: "5 min read", 
      rating: 4.9
    },
    {
      title: "Understanding methodology breakdown",
      description: "How our five analysis methods work together",
      readTime: "7 min read",
      rating: 4.7
    },
    {
      title: "Responsible gaming practices",
      description: "Playing safely and within your means",
      readTime: "4 min read",
      rating: 4.9
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
            <div className="flex items-center justify-center mb-6">
              <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-pink-500 rounded-lg flex items-center justify-center mr-4">
                <span className="text-white text-2xl">🔮</span>
              </div>
              <div>
                <h1 className="text-4xl md:text-5xl font-bold text-white">PatternSight</h1>
                <p className="text-xl text-orange-400">Help Center</p>
              </div>
            </div>
            <p className="text-2xl text-gray-300 mb-8">
              Find answers to your questions
            </p>

            {/* Search Bar */}
            <div className="max-w-2xl mx-auto relative">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  type="text"
                  placeholder="Search for help articles..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-4 py-4 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                />
                <button className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-orange-500 to-pink-500 text-white px-4 py-2 rounded-md hover:from-orange-600 hover:to-pink-600 transition-all duration-200">
                  🔍
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Help Topics */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Help Topics</h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 max-w-7xl mx-auto">
            {helpTopics.map((topic, index) => (
              <motion.div
                key={topic.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-all duration-200 cursor-pointer group"
              >
                <div className="p-6">
                  <div className={`w-16 h-16 bg-gradient-to-r ${topic.color} rounded-lg flex items-center justify-center mb-4 text-white group-hover:scale-110 transition-transform duration-200`}>
                    {topic.icon}
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2">{topic.title}</h3>
                  <p className="text-gray-300 mb-4">{topic.description}</p>
                  
                  <div className="space-y-2">
                    {topic.articles.map((article) => (
                      <div key={article} className="flex items-center text-gray-400 hover:text-white transition-colors duration-200">
                        <ChevronRight className="w-4 h-4 mr-2" />
                        <span className="text-sm">{article}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Links */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Quick Links</h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {quickLinks.map((link, index) => (
              <motion.a
                key={link.title}
                href={link.href}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-gradient-to-r from-purple-500/20 to-indigo-500/20 rounded-lg p-6 border border-purple-500/30 hover:from-purple-500/30 hover:to-indigo-500/30 transition-all duration-200 text-center group"
              >
                <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-pink-500 rounded-lg flex items-center justify-center mx-auto mb-4 text-white group-hover:scale-110 transition-transform duration-200">
                  {link.icon}
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{link.title}</h3>
                <p className="text-gray-300">{link.description}</p>
              </motion.a>
            ))}
          </div>
        </div>
      </section>

      {/* Getting Started Guide */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl font-bold text-white mb-6">Welcome to PatternSight</h2>
              <p className="text-xl text-gray-300">
                PatternSight is an entertainment platform that combines advanced mathematics, 
                artificial intelligence, and cosmic wisdom to analyze lottery patterns.
              </p>
              <div className="mt-6 p-4 bg-orange-500/20 rounded-lg border border-orange-500/30">
                <p className="text-orange-200">
                  <strong>Important:</strong> Our sophisticated algorithms provide predictions for entertainment purposes only.
                </p>
              </div>
            </motion.div>

            <div className="space-y-12">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="bg-white/5 rounded-lg p-8 border border-white/10"
              >
                <h3 className="text-2xl font-bold text-white mb-4 flex items-center">
                  <User className="w-6 h-6 mr-3 text-orange-400" />
                  Creating Your Account
                </h3>
                <div className="space-y-4 text-gray-300">
                  <p>To get started, click "Start Free" on our homepage.</p>
                  <p>Choose your subscription plan, provide your email address, and create a secure password.</p>
                  <p>You'll need to agree to our terms and complete the responsible gaming disclaimer.</p>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="bg-white/5 rounded-lg p-8 border border-white/10"
              >
                <h3 className="text-2xl font-bold text-white mb-4 flex items-center">
                  <Star className="w-6 h-6 mr-3 text-orange-400" />
                  Understanding Subscription Plans
                </h3>
                <div className="space-y-4 text-gray-300">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-blue-500/20 p-4 rounded-lg border border-blue-500/30">
                      <h4 className="font-bold text-white mb-2">Starter ($9.99 per play)</h4>
                      <p className="text-sm">Single prediction access.</p>
                    </div>
                    <div className="bg-purple-500/20 p-4 rounded-lg border border-purple-500/30">
                      <h4 className="font-bold text-white mb-2">Plus ($19.99/month)</h4>
                      <p className="text-sm">Multiple predictions and detailed analysis.</p>
                    </div>
                    <div className="bg-orange-500/20 p-4 rounded-lg border border-orange-500/30">
                      <h4 className="font-bold text-white mb-2">Pro ($29.99/month)</h4>
                      <p className="text-sm">Full access to all features, unlimited predictions, and premium cosmic insights.</p>
                    </div>
                  </div>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="bg-white/5 rounded-lg p-8 border border-white/10"
              >
                <h3 className="text-2xl font-bold text-white mb-4 flex items-center">
                  <Target className="w-6 h-6 mr-3 text-orange-400" />
                  Your First Prediction
                </h3>
                <div className="space-y-4 text-gray-300">
                  <p>After logging in, visit your dashboard and click "Generate Oracle Predictions."</p>
                  <p>Our AI will analyze current cosmic conditions and generate personalized predictions with confidence scores and detailed methodology breakdowns.</p>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </section>

      {/* Popular Articles */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">
              📈 Popular Articles
            </h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-6xl mx-auto">
            {popularArticles.map((article, index) => (
              <motion.div
                key={article.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-6 border border-white/10 hover:bg-white/10 transition-all duration-200 cursor-pointer group"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center">
                    <BookOpen className="w-5 h-5 text-orange-400 mr-2" />
                    <span className="text-sm text-gray-400">{article.readTime}</span>
                  </div>
                  <div className="flex items-center">
                    <Star className="w-4 h-4 text-yellow-400 mr-1" />
                    <span className="text-sm text-gray-400">{article.rating}</span>
                  </div>
                </div>
                <h3 className="text-xl font-bold text-white mb-2 group-hover:text-orange-400 transition-colors duration-200">
                  {article.title}
                </h3>
                <p className="text-gray-300">{article.description}</p>
                <div className="mt-4 flex items-center text-orange-400 group-hover:text-orange-300 transition-colors duration-200">
                  <span className="text-sm">Read more</span>
                  <ChevronRight className="w-4 h-4 ml-1" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

