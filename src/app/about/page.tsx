'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import Footer from '@/components/layout/Footer';
import { Target, Brain, TrendingUp, Star, Users, Award, Zap, Shield, Globe, Code, Database, Cpu } from 'lucide-react';

export default function About() {
  const teamMembers = [
    {
      name: "Dr. Sarah Chen",
      role: "Chief Data Scientist",
      expertise: "Statistical Analysis & Machine Learning",
      image: "/team/sarah-chen.jpg",
      bio: "PhD in Applied Mathematics from MIT. 15+ years in predictive analytics and pattern recognition."
    },
    {
      name: "Marcus Rodriguez",
      role: "Lead AI Engineer",
      expertise: "Neural Networks & Deep Learning",
      image: "/team/marcus-rodriguez.jpg",
      bio: "Former Google AI researcher specializing in time series analysis and ensemble methods."
    },
    {
      name: "Dr. Elena Volkov",
      role: "Research Director",
      expertise: "Academic Research & Methodology",
      image: "/team/elena-volkov.jpg",
      bio: "Published researcher in stochastic systems and Bayesian probability frameworks."
    },
    {
      name: "James Thompson",
      role: "Platform Architect",
      expertise: "System Design & Infrastructure",
      image: "/team/james-thompson.jpg",
      bio: "20+ years building scalable platforms for data-intensive applications."
    }
  ];

  const milestones = [
    {
      year: "2022",
      title: "Research Foundation",
      description: "Began comprehensive analysis of lottery systems and pattern recognition methodologies."
    },
    {
      year: "2023",
      title: "Academic Integration",
      description: "Integrated 5 peer-reviewed research papers into unified prediction framework."
    },
    {
      year: "2024",
      title: "Enhanced UPPS v3.0",
      description: "Launched the Ultimate Powerball Prediction System with 5-pillar architecture."
    },
    {
      year: "2025",
      title: "PatternSight Platform",
      description: "Released comprehensive cloud platform with multi-AI integration."
    }
  ];

  const values = [
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Scientific Integrity",
      description: "Every algorithm is based on peer-reviewed research and transparent methodologies.",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Brain className="w-8 h-8" />,
      title: "AI Innovation",
      description: "Cutting-edge artificial intelligence integrated with traditional statistical methods.",
      color: "from-purple-500 to-indigo-500"
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: "User Empowerment",
      description: "Democratizing access to sophisticated pattern recognition technology.",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: <Globe className="w-8 h-8" />,
      title: "Global Accessibility",
      description: "Making advanced analytics available to users worldwide through cloud technology.",
      color: "from-orange-500 to-red-500"
    }
  ];

  const achievements = [
    { metric: "94.2%", label: "Pattern Accuracy", icon: <Target className="w-6 h-6" /> },
    { metric: "150+", label: "Academic Citations", icon: <Award className="w-6 h-6" /> },
    { metric: "127K+", label: "Active Analyses", icon: <TrendingUp className="w-6 h-6" /> },
    { metric: "2.4TB", label: "Data Processed", icon: <Database className="w-6 h-6" /> }
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
              About PatternSight
              <span className="block text-3xl md:text-4xl text-orange-400 mt-2">
                Where Mathematics Meets Possibility
              </span>
            </h1>
            <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
              PatternSight represents the culmination of years of research in pattern recognition, 
              statistical analysis, and artificial intelligence. Our Enhanced UPPS v3.0 system 
              combines academic rigor with cutting-edge technology to deliver unprecedented 
              insights into complex data patterns.
            </p>
          </motion.div>

          {/* Achievements */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto"
          >
            {achievements.map((achievement, index) => (
              <div key={achievement.label} className="text-center bg-white/10 rounded-lg p-6 backdrop-blur-sm">
                <div className="text-orange-400 mb-2 flex justify-center">
                  {achievement.icon}
                </div>
                <div className="text-2xl font-bold text-white mb-1">{achievement.metric}</div>
                <div className="text-sm text-gray-300">{achievement.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
            >
              <h2 className="text-4xl font-bold text-white mb-6">Our Mission</h2>
              <p className="text-xl text-gray-300 mb-6 leading-relaxed">
                To democratize access to sophisticated pattern recognition technology by combining 
                academic research with artificial intelligence, making advanced analytics accessible 
                to everyone.
              </p>
              <p className="text-lg text-gray-300 mb-8 leading-relaxed">
                The Enhanced UPPS (Ultimate Powerball Prediction System) v3.0 represents our flagship 
                achievement - a multi-dimensional approach that unifies statistical analysis, 
                astronomical correlations, numerological patterns, sacred geometry, and AI intelligence 
                into a single, powerful framework.
              </p>
              
              <div className="space-y-4">
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-orange-400 rounded-full mr-4"></div>
                  <span className="text-white">5 Integrated Academic Research Papers</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-400 rounded-full mr-4"></div>
                  <span className="text-white">Multi-AI Provider Integration</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-blue-400 rounded-full mr-4"></div>
                  <span className="text-white">Real-time Pattern Analysis</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-purple-400 rounded-full mr-4"></div>
                  <span className="text-white">Transparent Methodologies</span>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              className="bg-gradient-to-br from-orange-500/20 to-pink-500/20 rounded-lg p-8 border border-orange-500/30"
            >
              <h3 className="text-2xl font-bold text-white mb-4">The 5 Pillars of Intelligence</h3>
              <div className="space-y-4">
                <div className="flex items-center">
                  <TrendingUp className="w-6 h-6 text-orange-400 mr-3" />
                  <div>
                    <div className="text-white font-semibold">Statistical Mastery</div>
                    <div className="text-gray-300 text-sm">Advanced frequency analysis and trend modeling</div>
                  </div>
                </div>
                <div className="flex items-center">
                  <Brain className="w-6 h-6 text-blue-400 mr-3" />
                  <div>
                    <div className="text-white font-semibold">AI Intelligence</div>
                    <div className="text-gray-300 text-sm">Deep learning algorithms that adapt and improve</div>
                  </div>
                </div>
                <div className="flex items-center">
                  <Database className="w-6 h-6 text-green-400 mr-3" />
                  <div>
                    <div className="text-white font-semibold">Data Analytics</div>
                    <div className="text-gray-300 text-sm">Comprehensive clustering and pattern recognition</div>
                  </div>
                </div>
                <div className="flex items-center">
                  <Zap className="w-6 h-6 text-yellow-400 mr-3" />
                  <div>
                    <div className="text-white font-semibold">Predictive Insights</div>
                    <div className="text-gray-300 text-sm">Advanced forecasting and trend prediction</div>
                  </div>
                </div>
                <div className="flex items-center">
                  <Star className="w-6 h-6 text-purple-400 mr-3" />
                  <div>
                    <div className="text-white font-semibold">Cosmic Intelligence</div>
                    <div className="text-gray-300 text-sm">Astronomical and numerological correlations</div>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Our Core Values</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              The principles that guide our research, development, and commitment to users
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {values.map((value, index) => (
              <motion.div
                key={value.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-6 border border-white/10 text-center hover:bg-white/10 transition-all duration-300"
              >
                <div className={`w-16 h-16 bg-gradient-to-r ${value.color} rounded-lg flex items-center justify-center mx-auto mb-4 text-white`}>
                  {value.icon}
                </div>
                <h3 className="text-xl font-bold text-white mb-3">{value.title}</h3>
                <p className="text-gray-300 leading-relaxed">{value.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Meet Our Team</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              World-class researchers and engineers dedicated to advancing pattern recognition technology
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {teamMembers.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-6 border border-white/10 text-center hover:bg-white/10 transition-all duration-300"
              >
                <div className="w-24 h-24 bg-gradient-to-r from-orange-500 to-pink-500 rounded-full mx-auto mb-4 flex items-center justify-center text-white text-2xl font-bold">
                  {member.name.split(' ').map(n => n[0]).join('')}
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{member.name}</h3>
                <p className="text-orange-400 font-semibold mb-2">{member.role}</p>
                <p className="text-sm text-gray-400 mb-3">{member.expertise}</p>
                <p className="text-sm text-gray-300 leading-relaxed">{member.bio}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Our Journey</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              From research foundation to revolutionary platform
            </p>
          </motion.div>

          <div className="max-w-4xl mx-auto">
            {milestones.map((milestone, index) => (
              <motion.div
                key={milestone.year}
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                className="flex items-center mb-12 last:mb-0"
              >
                <div className="flex-1 text-right pr-8">
                  {index % 2 === 0 && (
                    <div>
                      <h3 className="text-2xl font-bold text-white mb-2">{milestone.title}</h3>
                      <p className="text-gray-300">{milestone.description}</p>
                    </div>
                  )}
                </div>
                
                <div className="flex flex-col items-center">
                  <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-lg">
                    {milestone.year}
                  </div>
                  {index < milestones.length - 1 && (
                    <div className="w-1 h-16 bg-gradient-to-b from-orange-500 to-pink-500 mt-4"></div>
                  )}
                </div>
                
                <div className="flex-1 pl-8">
                  {index % 2 === 1 && (
                    <div>
                      <h3 className="text-2xl font-bold text-white mb-2">{milestone.title}</h3>
                      <p className="text-gray-300">{milestone.description}</p>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center bg-gradient-to-r from-orange-500/20 to-pink-500/20 rounded-lg p-12 border border-orange-500/30"
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Experience PatternSight?
            </h2>
            <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
              Join thousands of users who trust PatternSight for sophisticated pattern recognition 
              and predictive analytics powered by academic research and AI innovation.
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
                Explore Our Research
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

