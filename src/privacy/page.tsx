'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Footer from '@/components/layout/Footer';
import { Shield, Lock, Eye, Database, Globe, UserCheck } from 'lucide-react';

export default function PrivacyPolicy() {
  const lastUpdated = "September 3, 2025";

  const sections = [
    {
      title: "Information We Collect",
      icon: <Database className="w-6 h-6" />,
      content: [
        {
          subtitle: "Account Information",
          details: "When you create a PatternSight account, we collect your email address, name, and subscription preferences. This information is necessary to provide our services and manage your account."
        },
        {
          subtitle: "Usage Data",
          details: "We collect information about how you use our platform, including prediction requests, analysis preferences, and feature usage. This helps us improve our Enhanced UPPS v3.0 system and provide better recommendations."
        },
        {
          subtitle: "Technical Information",
          details: "We automatically collect technical information such as IP addresses, browser type, device information, and access times. This data is used for security, analytics, and platform optimization."
        },
        {
          subtitle: "Prediction Data",
          details: "We store your prediction history, analysis results, and preferences to provide personalized insights and improve our algorithms. All prediction data is encrypted and securely stored."
        }
      ]
    },
    {
      title: "How We Use Your Information",
      icon: <UserCheck className="w-6 h-6" />,
      content: [
        {
          subtitle: "Service Provision",
          details: "We use your information to provide PatternSight services, including generating predictions, analyzing patterns, and delivering personalized insights through our Enhanced UPPS v3.0 system."
        },
        {
          subtitle: "Account Management",
          details: "Your account information is used to manage subscriptions, process payments, provide customer support, and communicate important service updates."
        },
        {
          subtitle: "Platform Improvement",
          details: "We analyze usage patterns and feedback to enhance our algorithms, develop new features, and improve the overall user experience. All analysis is performed on aggregated, anonymized data."
        },
        {
          subtitle: "Security and Fraud Prevention",
          details: "We use your information to detect and prevent fraudulent activities, protect against security threats, and ensure the integrity of our platform."
        }
      ]
    },
    {
      title: "Information Sharing",
      icon: <Globe className="w-6 h-6" />,
      content: [
        {
          subtitle: "Third-Party Services",
          details: "We work with trusted third-party providers for payment processing (Stripe), analytics (anonymized), and infrastructure services (Supabase, Vercel). These partners are bound by strict data protection agreements."
        },
        {
          subtitle: "Legal Requirements",
          details: "We may disclose information when required by law, court order, or government request, or when necessary to protect our rights, users' safety, or investigate fraud."
        },
        {
          subtitle: "Business Transfers",
          details: "In the event of a merger, acquisition, or sale of assets, user information may be transferred as part of the transaction, subject to the same privacy protections."
        },
        {
          subtitle: "Aggregated Data",
          details: "We may share aggregated, anonymized statistics about platform usage, prediction accuracy, and trends for research and marketing purposes. This data cannot be used to identify individual users."
        }
      ]
    },
    {
      title: "Data Security",
      icon: <Lock className="w-6 h-6" />,
      content: [
        {
          subtitle: "Encryption",
          details: "All data is encrypted in transit using TLS 1.3 and at rest using AES-256 encryption. Our database connections use SSL/TLS encryption for additional security."
        },
        {
          subtitle: "Access Controls",
          details: "We implement strict access controls, multi-factor authentication, and role-based permissions to ensure only authorized personnel can access user data."
        },
        {
          subtitle: "Regular Audits",
          details: "Our security practices are regularly audited and updated to meet industry standards. We conduct penetration testing and vulnerability assessments quarterly."
        },
        {
          subtitle: "Data Backup",
          details: "We maintain secure, encrypted backups of user data with geographic redundancy to ensure data availability and disaster recovery capabilities."
        }
      ]
    },
    {
      title: "Your Rights",
      icon: <Eye className="w-6 h-6" />,
      content: [
        {
          subtitle: "Access and Portability",
          details: "You have the right to access your personal data and request a copy in a machine-readable format. You can export your prediction history and account data from your dashboard."
        },
        {
          subtitle: "Correction and Updates",
          details: "You can update your account information, preferences, and settings at any time through your dashboard. We encourage keeping your information current and accurate."
        },
        {
          subtitle: "Deletion",
          details: "You can request deletion of your account and associated data. Some information may be retained for legal compliance, security, or legitimate business purposes as outlined in our retention policy."
        },
        {
          subtitle: "Opt-Out",
          details: "You can opt out of non-essential communications, analytics tracking, and certain data processing activities. Essential service communications cannot be disabled while maintaining an active account."
        }
      ]
    },
    {
      title: "Cookies and Tracking",
      icon: <Shield className="w-6 h-6" />,
      content: [
        {
          subtitle: "Essential Cookies",
          details: "We use essential cookies for authentication, security, and basic platform functionality. These cookies are necessary for the service to work properly."
        },
        {
          subtitle: "Analytics Cookies",
          details: "We use analytics cookies to understand how users interact with our platform, identify popular features, and improve user experience. You can opt out of analytics tracking."
        },
        {
          subtitle: "Preference Cookies",
          details: "We store your preferences, settings, and customizations in cookies to provide a personalized experience across sessions."
        },
        {
          subtitle: "Third-Party Cookies",
          details: "Some third-party services may set cookies for payment processing, support chat, or other integrated features. These are governed by their respective privacy policies."
        }
      ]
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
              Privacy Policy
              <span className="block text-3xl md:text-4xl text-orange-400 mt-2">
                Your Data, Your Rights
              </span>
            </h1>
            <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
              At PatternSight, we are committed to protecting your privacy and ensuring transparency 
              about how we collect, use, and safeguard your personal information. This policy explains 
              our practices in clear, understandable terms.
            </p>
            <div className="mt-8 text-gray-400">
              <p>Last updated: {lastUpdated}</p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Privacy Principles */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Our Privacy Principles</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              These core principles guide how we handle your data
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="bg-white/5 rounded-lg p-6 border border-white/10 text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center mx-auto mb-4 text-white">
                <Shield className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Transparency</h3>
              <p className="text-gray-300">
                We clearly explain what data we collect, how we use it, and who we share it with. 
                No hidden practices or unclear terms.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="bg-white/5 rounded-lg p-6 border border-white/10 text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center mx-auto mb-4 text-white">
                <Lock className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Security</h3>
              <p className="text-gray-300">
                Your data is protected with enterprise-grade security measures, including encryption, 
                access controls, and regular security audits.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="bg-white/5 rounded-lg p-6 border border-white/10 text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center mx-auto mb-4 text-white">
                <UserCheck className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Control</h3>
              <p className="text-gray-300">
                You have full control over your data with rights to access, correct, delete, 
                and export your information at any time.
              </p>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Detailed Sections */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto space-y-12">
            {sections.map((section, index) => (
              <motion.div
                key={section.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg border border-white/10"
              >
                <div className="p-6 border-b border-white/10">
                  <div className="flex items-center">
                    <div className="text-orange-400 mr-4">
                      {section.icon}
                    </div>
                    <h3 className="text-2xl font-bold text-white">{section.title}</h3>
                  </div>
                </div>

                <div className="p-6 space-y-6">
                  {section.content.map((item, idx) => (
                    <div key={item.subtitle}>
                      <h4 className="text-lg font-semibold text-white mb-2">{item.subtitle}</h4>
                      <p className="text-gray-300 leading-relaxed">{item.details}</p>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center bg-gradient-to-r from-orange-500/20 to-pink-500/20 rounded-lg p-12 border border-orange-500/30 max-w-4xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Questions About Your Privacy?
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              We're here to help you understand how we protect your data and respect your privacy rights.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="text-center">
                <h4 className="text-white font-semibold mb-2">Privacy Officer</h4>
                <p className="text-gray-300">privacy@patternsight.ai</p>
              </div>
              <div className="text-center">
                <h4 className="text-white font-semibold mb-2">Data Protection</h4>
                <p className="text-gray-300">dpo@patternsight.ai</p>
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="/contact"
                className="px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200"
              >
                Contact Us
              </a>
              <a
                href="/cookies"
                className="px-8 py-4 bg-white/10 text-white rounded-lg font-semibold hover:bg-white/20 transition-all duration-200 border border-white/20"
              >
                Cookie Policy
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

