'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Footer from '@/components/layout/Footer';
import { Scale, FileText, Shield, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

export default function TermsOfService() {
  const lastUpdated = "September 3, 2025";

  const sections = [
    {
      title: "Acceptance of Terms",
      icon: <CheckCircle className="w-6 h-6" />,
      content: [
        {
          subtitle: "Agreement to Terms",
          details: "By accessing and using PatternSight, you accept and agree to be bound by the terms and provision of this agreement. If you do not agree to abide by the above, please do not use this service."
        },
        {
          subtitle: "Modifications",
          details: "We reserve the right to modify these terms at any time. Changes will be effective immediately upon posting. Your continued use of the service constitutes acceptance of the modified terms."
        },
        {
          subtitle: "Eligibility",
          details: "You must be at least 18 years old to use PatternSight. By using our service, you represent and warrant that you meet this age requirement and have the legal capacity to enter into this agreement."
        }
      ]
    },
    {
      title: "Service Description",
      icon: <FileText className="w-6 h-6" />,
      content: [
        {
          subtitle: "Enhanced UPPS v3.0 System",
          details: "PatternSight provides pattern recognition and predictive analytics through our Enhanced Ultimate Powerball Prediction System v3.0, which integrates 5 academic research papers and artificial intelligence technologies."
        },
        {
          subtitle: "Predictions and Analysis",
          details: "Our service generates predictions and analysis based on statistical models, AI algorithms, and pattern recognition. These are for informational and entertainment purposes only and do not guarantee any outcomes."
        },
        {
          subtitle: "Academic Integration",
          details: "Our system incorporates peer-reviewed research including Compound-Dirichlet-Multinomial models, Order Statistics Theory, Statistical-Neural Hybrid analysis, XGBoost Behavioral modeling, and Deep Learning Time Series analysis."
        },
        {
          subtitle: "Service Availability",
          details: "We strive to maintain 99.9% uptime but do not guarantee uninterrupted service. Maintenance, updates, and unforeseen circumstances may temporarily affect availability."
        }
      ]
    },
    {
      title: "User Responsibilities",
      icon: <Shield className="w-6 h-6" />,
      content: [
        {
          subtitle: "Account Security",
          details: "You are responsible for maintaining the confidentiality of your account credentials and for all activities that occur under your account. Notify us immediately of any unauthorized use."
        },
        {
          subtitle: "Lawful Use",
          details: "You agree to use PatternSight only for lawful purposes and in accordance with these terms. You may not use our service for any illegal activities or to violate any applicable laws or regulations."
        },
        {
          subtitle: "Prohibited Activities",
          details: "You may not attempt to reverse engineer, hack, or compromise our systems; share your account with others; use automated tools to access our service; or engage in any activity that could harm our platform or other users."
        },
        {
          subtitle: "Content and Conduct",
          details: "You are responsible for any content you submit and must ensure it does not violate any laws, infringe on rights of others, or contain harmful, offensive, or inappropriate material."
        }
      ]
    },
    {
      title: "Subscription and Billing",
      icon: <Scale className="w-6 h-6" />,
      content: [
        {
          subtitle: "Subscription Tiers",
          details: "PatternSight offers multiple subscription tiers: Oracle Starter ($3.99), Cosmic Oracle Pro ($19.99), and Cosmic Oracle Elite ($49.99). Each tier includes different features and prediction limits."
        },
        {
          subtitle: "Billing and Payments",
          details: "Subscriptions are billed monthly in advance. All fees are non-refundable except as required by law. We use secure third-party payment processors and do not store your payment information."
        },
        {
          subtitle: "Cancellation",
          details: "You may cancel your subscription at any time through your account settings. Cancellation will take effect at the end of your current billing period. You will retain access to paid features until the end of the paid period."
        },
        {
          subtitle: "Price Changes",
          details: "We reserve the right to modify subscription prices with 30 days advance notice. Price changes will not affect your current billing period but will apply to subsequent renewals."
        }
      ]
    },
    {
      title: "Intellectual Property",
      icon: <FileText className="w-6 h-6" />,
      content: [
        {
          subtitle: "Our Intellectual Property",
          details: "PatternSight, Enhanced UPPS v3.0, our algorithms, software, content, and trademarks are owned by us or our licensors. You may not copy, modify, distribute, or create derivative works without permission."
        },
        {
          subtitle: "Academic Research",
          details: "Our system incorporates published academic research under fair use and proper attribution. We respect the intellectual property rights of researchers and maintain proper citations."
        },
        {
          subtitle: "User Content",
          details: "You retain ownership of content you submit but grant us a license to use, modify, and display it as necessary to provide our services. You represent that you have the right to grant this license."
        },
        {
          subtitle: "DMCA Compliance",
          details: "We respect intellectual property rights and comply with the Digital Millennium Copyright Act. If you believe your copyright has been infringed, please contact us with a proper DMCA notice."
        }
      ]
    },
    {
      title: "Disclaimers and Limitations",
      icon: <AlertTriangle className="w-6 h-6" />,
      content: [
        {
          subtitle: "No Guarantees",
          details: "PatternSight provides predictions and analysis for informational purposes only. We make no guarantees about accuracy, completeness, or outcomes. Past performance does not indicate future results."
        },
        {
          subtitle: "Entertainment Purpose",
          details: "Our service is intended for entertainment and educational purposes. Any predictions or analysis should not be considered as professional advice or guarantees of future outcomes."
        },
        {
          subtitle: "Limitation of Liability",
          details: "To the maximum extent permitted by law, we shall not be liable for any indirect, incidental, special, consequential, or punitive damages, including but not limited to loss of profits, data, or use."
        },
        {
          subtitle: "Service 'As Is'",
          details: "PatternSight is provided 'as is' and 'as available' without warranties of any kind, either express or implied, including but not limited to merchantability, fitness for a particular purpose, or non-infringement."
        }
      ]
    },
    {
      title: "Privacy and Data",
      icon: <Shield className="w-6 h-6" />,
      content: [
        {
          subtitle: "Privacy Policy",
          details: "Your privacy is important to us. Our Privacy Policy explains how we collect, use, and protect your information. By using our service, you agree to our privacy practices."
        },
        {
          subtitle: "Data Security",
          details: "We implement industry-standard security measures to protect your data, including encryption, access controls, and regular security audits. However, no system is completely secure."
        },
        {
          subtitle: "Data Retention",
          details: "We retain your data as necessary to provide services, comply with legal obligations, and improve our platform. You can request data deletion subject to our retention policies."
        },
        {
          subtitle: "Third-Party Services",
          details: "We use trusted third-party services for payments, analytics, and infrastructure. These partners are bound by strict data protection agreements and privacy standards."
        }
      ]
    },
    {
      title: "Termination",
      icon: <XCircle className="w-6 h-6" />,
      content: [
        {
          subtitle: "Termination by You",
          details: "You may terminate your account at any time by canceling your subscription and deleting your account through your dashboard settings. Some data may be retained as outlined in our Privacy Policy."
        },
        {
          subtitle: "Termination by Us",
          details: "We may suspend or terminate your account if you violate these terms, engage in fraudulent activity, or for any reason with reasonable notice. We will provide refunds as required by law."
        },
        {
          subtitle: "Effect of Termination",
          details: "Upon termination, your right to use PatternSight ceases immediately. We may delete your account data after a reasonable period, subject to legal retention requirements."
        },
        {
          subtitle: "Survival",
          details: "Provisions regarding intellectual property, disclaimers, limitations of liability, and dispute resolution will survive termination of this agreement."
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
              Terms of Service
              <span className="block text-3xl md:text-4xl text-orange-400 mt-2">
                Your Agreement with PatternSight
              </span>
            </h1>
            <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
              These terms govern your use of PatternSight and our Enhanced UPPS v3.0 system. 
              Please read them carefully as they contain important information about your rights 
              and responsibilities when using our pattern recognition platform.
            </p>
            <div className="mt-8 text-gray-400">
              <p>Last updated: {lastUpdated}</p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Key Points */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-6">Key Points</h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Important highlights from our terms of service
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
                <FileText className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Entertainment Purpose</h3>
              <p className="text-gray-300">
                PatternSight provides predictions and analysis for entertainment and educational 
                purposes only. No guarantees are made about outcomes or accuracy.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="bg-white/5 rounded-lg p-6 border border-white/10 text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center mx-auto mb-4 text-white">
                <Shield className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Your Responsibilities</h3>
              <p className="text-gray-300">
                You're responsible for account security, lawful use of our platform, 
                and compliance with these terms and applicable laws.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="bg-white/5 rounded-lg p-6 border border-white/10 text-center"
            >
              <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center mx-auto mb-4 text-white">
                <Scale className="w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Fair Usage</h3>
              <p className="text-gray-300">
                Our subscription tiers provide different prediction limits and features. 
                Cancellation takes effect at the end of your billing period.
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

      {/* Important Notice */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-4xl mx-auto"
          >
            <div className="bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded-lg p-8 border border-orange-500/30">
              <div className="flex items-start">
                <AlertTriangle className="w-8 h-8 text-orange-400 mr-4 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="text-2xl font-bold text-white mb-4">Important Disclaimer</h3>
                  <div className="space-y-4 text-gray-300">
                    <p className="leading-relaxed">
                      <strong className="text-white">Entertainment and Educational Use Only:</strong> PatternSight 
                      and our Enhanced UPPS v3.0 system are designed for entertainment and educational purposes. 
                      Our predictions and analysis are based on statistical models and AI algorithms but do not 
                      guarantee any outcomes.
                    </p>
                    <p className="leading-relaxed">
                      <strong className="text-white">No Professional Advice:</strong> Nothing provided by 
                      PatternSight constitutes professional, financial, or gambling advice. Users should not 
                      rely solely on our predictions for any decision-making purposes.
                    </p>
                    <p className="leading-relaxed">
                      <strong className="text-white">Responsible Use:</strong> Please use our platform responsibly 
                      and within your means. If you have concerns about gambling or addictive behaviors, please 
                      seek appropriate help and support.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Contact Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center bg-gradient-to-r from-purple-500/20 to-indigo-500/20 rounded-lg p-12 border border-purple-500/30 max-w-4xl mx-auto"
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Questions About These Terms?
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              We're here to help clarify any questions you may have about our terms of service.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
              <div className="text-center">
                <h4 className="text-white font-semibold mb-2">Legal Team</h4>
                <p className="text-gray-300">legal@patternsight.ai</p>
              </div>
              <div className="text-center">
                <h4 className="text-white font-semibold mb-2">General Support</h4>
                <p className="text-gray-300">support@patternsight.ai</p>
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="/contact"
                className="px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200"
              >
                Contact Support
              </a>
              <a
                href="/privacy"
                className="px-8 py-4 bg-white/10 text-white rounded-lg font-semibold hover:bg-white/20 transition-all duration-200 border border-white/20"
              >
                Privacy Policy
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

