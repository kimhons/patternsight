'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Footer from '@/components/layout/Footer';
import { CheckCircle, Crown, Star, Zap, Shield, Users, Clock, Infinity } from 'lucide-react';

export default function Pricing() {
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'yearly'>('monthly');

  const subscriptionTiers = [
    {
      id: 'starter',
      name: 'Oracle Starter',
      description: 'Perfect for beginners exploring pattern recognition',
      monthlyPrice: 3.99,
      yearlyPrice: 39.99,
      predictions: '3 per day',
      features: [
        'Basic pattern analysis',
        'Daily cosmic insights',
        'Community access',
        'Enhanced UPPS v3.0 access',
        'Email support',
        'Mobile app access'
      ],
      color: 'from-blue-500 to-cyan-500',
      icon: <Star className="w-6 h-6" />,
      popular: false
    },
    {
      id: 'pro',
      name: 'Cosmic Oracle Pro',
      description: 'Advanced features for serious pattern analysts',
      monthlyPrice: 19.99,
      yearlyPrice: 199.99,
      predictions: '25 per day',
      features: [
        'Advanced AI analysis',
        'Cosmic intelligence',
        'Priority support',
        'All 5 academic pillars',
        'Detailed explanations',
        'API access (limited)',
        'Custom reports',
        'Advanced analytics'
      ],
      color: 'from-orange-500 to-pink-500',
      icon: <Crown className="w-6 h-6" />,
      popular: true
    },
    {
      id: 'elite',
      name: 'Cosmic Oracle Elite',
      description: 'Maximum power for professional users',
      monthlyPrice: 49.99,
      yearlyPrice: 499.99,
      predictions: '50 per day',
      features: [
        'Maximum AI power',
        'Personal cosmic profile',
        'VIP support',
        'Full API access',
        'White-label options',
        'Advanced analytics',
        'Custom AI models',
        'Dedicated account manager'
      ],
      color: 'from-purple-500 to-indigo-500',
      icon: <Infinity className="w-6 h-6" />,
      popular: false
    }
  ];

  const predictionBundles = [
    {
      predictions: 10,
      originalPrice: 4.99,
      salePrice: 2.49,
      perPrediction: 0.25,
      popular: false
    },
    {
      predictions: 20,
      originalPrice: 8.99,
      salePrice: 4.49,
      perPrediction: 0.22,
      popular: false
    },
    {
      predictions: 50,
      originalPrice: 19.99,
      salePrice: 9.99,
      perPrediction: 0.20,
      popular: true
    },
    {
      predictions: 100,
      originalPrice: 34.99,
      salePrice: 17.49,
      perPrediction: 0.17,
      popular: false
    }
  ];

  const enterpriseFeatures = [
    'Unlimited predictions',
    'Custom AI model training',
    'Dedicated infrastructure',
    'SLA guarantees',
    'On-premise deployment',
    'Custom integrations',
    'Advanced security',
    'Training and onboarding'
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
              Choose Your
              <span className="block text-orange-400">Intelligence Level</span>
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Unlock the power of Enhanced UPPS v3.0 Academic Integration System with 
              flexible pricing options designed for every level of pattern analysis needs.
            </p>
          </motion.div>

          {/* Billing Toggle */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="flex justify-center mb-12"
          >
            <div className="bg-white/10 rounded-lg p-1 backdrop-blur-sm">
              <button
                onClick={() => setBillingCycle('monthly')}
                className={`px-6 py-2 rounded-md transition-all duration-200 ${
                  billingCycle === 'monthly'
                    ? 'bg-orange-500 text-white'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                Monthly
              </button>
              <button
                onClick={() => setBillingCycle('yearly')}
                className={`px-6 py-2 rounded-md transition-all duration-200 ${
                  billingCycle === 'yearly'
                    ? 'bg-orange-500 text-white'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                Yearly
                <span className="ml-2 text-xs bg-green-500 text-white px-2 py-1 rounded-full">
                  Save 17%
                </span>
              </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Subscription Tiers */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {subscriptionTiers.map((tier, index) => (
              <motion.div
                key={tier.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className={`relative bg-white/10 backdrop-blur-sm rounded-lg p-8 border-2 ${
                  tier.popular 
                    ? 'border-orange-500 ring-2 ring-orange-500/50' 
                    : 'border-white/20'
                } hover:bg-white/15 transition-all duration-300`}
              >
                {tier.popular && (
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                    <span className="bg-gradient-to-r from-orange-500 to-pink-500 text-white px-4 py-2 rounded-full text-sm font-bold">
                      MOST POPULAR
                    </span>
                  </div>
                )}

                <div className="text-center mb-6">
                  <div className={`w-16 h-16 bg-gradient-to-r ${tier.color} rounded-lg flex items-center justify-center mx-auto mb-4 text-white`}>
                    {tier.icon}
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-2">{tier.name}</h3>
                  <p className="text-gray-300 text-sm">{tier.description}</p>
                </div>

                <div className="text-center mb-6">
                  <div className="text-4xl font-bold text-white mb-2">
                    ${billingCycle === 'monthly' ? tier.monthlyPrice : tier.yearlyPrice}
                    {billingCycle === 'yearly' && (
                      <span className="text-lg text-gray-400 line-through ml-2">
                        ${(tier.monthlyPrice * 12).toFixed(2)}
                      </span>
                    )}
                  </div>
                  <p className="text-gray-400">
                    {billingCycle === 'monthly' ? 'per month' : 'per year'}
                  </p>
                  <p className="text-orange-400 font-semibold mt-1">{tier.predictions}</p>
                </div>

                <ul className="space-y-3 mb-8">
                  {tier.features.map((feature, idx) => (
                    <li key={idx} className="flex items-start text-gray-300">
                      <CheckCircle className="w-5 h-5 text-green-400 mr-3 mt-0.5 flex-shrink-0" />
                      {feature}
                    </li>
                  ))}
                </ul>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className={`w-full py-3 rounded-lg font-semibold transition-all duration-200 ${
                    tier.popular
                      ? 'bg-gradient-to-r from-orange-500 to-pink-500 text-white hover:from-orange-600 hover:to-pink-600'
                      : 'bg-white/10 text-white hover:bg-white/20 border border-white/20'
                  }`}
                >
                  {tier.popular ? 'Upgrade Now' : 'Get Started'}
                </motion.button>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Flash Sale Bundles */}
      <section className="py-16 bg-gradient-to-r from-pink-600 to-red-600">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              ⚡ FLASH SALE: Prediction Bundles
            </h2>
            <p className="text-xl text-white/90">
              50% OFF - Never expire, use anytime!
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            {predictionBundles.map((bundle, index) => (
              <motion.div
                key={bundle.predictions}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className={`bg-white/20 backdrop-blur-sm rounded-lg p-6 text-center ${
                  bundle.popular ? 'ring-2 ring-yellow-400' : ''
                }`}
              >
                {bundle.popular && (
                  <div className="bg-yellow-400 text-black px-3 py-1 rounded-full text-sm font-bold mb-4">
                    BEST VALUE
                  </div>
                )}
                
                <h3 className="text-2xl font-bold text-white mb-2">
                  {bundle.predictions} Predictions
                </h3>
                
                <div className="mb-4">
                  <div className="text-lg text-white/60 line-through">
                    ${bundle.originalPrice}
                  </div>
                  <div className="text-3xl font-bold text-white">
                    ${bundle.salePrice}
                  </div>
                  <div className="text-white/80 text-sm">
                    ${bundle.perPrediction.toFixed(2)} each
                  </div>
                </div>

                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="w-full py-3 bg-white text-pink-600 rounded-lg font-semibold hover:bg-gray-100 transition-all duration-200"
                >
                  Buy Now
                </motion.button>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Enterprise Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center mb-12"
            >
              <h2 className="text-4xl font-bold text-white mb-4">Enterprise Solutions</h2>
              <p className="text-xl text-gray-300">
                Custom solutions for large organizations and research institutions
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="bg-white/5 rounded-lg p-8 border border-white/10"
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-2xl font-bold text-white mb-4">Enterprise Features</h3>
                  <ul className="space-y-3">
                    {enterpriseFeatures.map((feature, index) => (
                      <li key={index} className="flex items-center text-gray-300">
                        <Shield className="w-5 h-5 text-green-400 mr-3" />
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="text-center lg:text-left">
                  <h3 className="text-2xl font-bold text-white mb-4">Ready to Scale?</h3>
                  <p className="text-gray-300 mb-6">
                    Get a custom quote tailored to your organization's needs. 
                    Our enterprise solutions include dedicated support, custom integrations, 
                    and advanced security features.
                  </p>
                  
                  <div className="space-y-4">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="w-full lg:w-auto px-8 py-3 bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-lg font-semibold hover:from-purple-600 hover:to-indigo-600 transition-all duration-200"
                    >
                      Contact Sales
                    </motion.button>
                    
                    <div className="text-sm text-gray-400">
                      <div className="flex items-center justify-center lg:justify-start">
                        <Users className="w-4 h-4 mr-2" />
                        Trusted by 500+ organizations
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-white mb-4">Frequently Asked Questions</h2>
          </motion.div>

          <div className="max-w-3xl mx-auto space-y-6">
            {[
              {
                question: "What is Enhanced UPPS v3.0?",
                answer: "Enhanced UPPS v3.0 is our Academic Integration System that combines 8 peer-reviewed research papers with advanced AI to provide the most accurate pattern recognition available."
              },
              {
                question: "How accurate are the predictions?",
                answer: "Our system achieves 94.2% pattern accuracy through the integration of academic research, statistical analysis, and multiple AI providers including OpenAI, Claude, and DeepSeek."
              },
              {
                question: "Can I upgrade or downgrade my plan?",
                answer: "Yes, you can change your subscription tier at any time. Changes take effect immediately, and we'll prorate any billing differences."
              },
              {
                question: "Do prediction bundles expire?",
                answer: "No, prediction bundles never expire. You can use them at your own pace, making them perfect for occasional users."
              }
            ].map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-6 border border-white/10"
              >
                <h3 className="text-lg font-bold text-white mb-2">{faq.question}</h3>
                <p className="text-gray-300">{faq.answer}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

