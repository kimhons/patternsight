'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Footer from '@/components/layout/Footer';
import { Cookie, Shield, Settings, Eye, Database, Clock } from 'lucide-react';

export default function CookiePolicy() {
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
                <Cookie className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-4xl md:text-5xl font-bold text-white">Cookie Policy</h1>
                <p className="text-xl text-orange-400">PatternSight</p>
              </div>
            </div>
            <p className="text-xl text-gray-300">
              How we use cookies to enhance your experience
            </p>
            <p className="text-sm text-gray-400 mt-4">
              Last updated: September 3, 2025
            </p>
          </motion.div>
        </div>
      </section>

      {/* Cookie Policy Content */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="space-y-12"
            >
              {/* What Are Cookies */}
              <div className="bg-white/5 rounded-lg p-8 border border-white/10">
                <div className="flex items-center mb-6">
                  <Cookie className="w-6 h-6 text-orange-400 mr-3" />
                  <h2 className="text-2xl font-bold text-white">What Are Cookies?</h2>
                </div>
                <div className="space-y-4 text-gray-300">
                  <p>
                    Cookies are small text files that are placed on your computer or mobile device when you visit our website. 
                    They are widely used to make websites work more efficiently and provide information to website owners.
                  </p>
                  <p>
                    PatternSight uses cookies to enhance your browsing experience, analyze site traffic, and personalize content 
                    to better serve our users' needs.
                  </p>
                </div>
              </div>

              {/* Types of Cookies */}
              <div className="bg-white/5 rounded-lg p-8 border border-white/10">
                <div className="flex items-center mb-6">
                  <Settings className="w-6 h-6 text-orange-400 mr-3" />
                  <h2 className="text-2xl font-bold text-white">Types of Cookies We Use</h2>
                </div>
                <div className="space-y-6">
                  <div className="bg-blue-500/20 p-4 rounded-lg border border-blue-500/30">
                    <h3 className="text-lg font-bold text-white mb-2">Essential Cookies</h3>
                    <p className="text-gray-300 text-sm">
                      These cookies are necessary for the website to function properly. They enable core functionality 
                      such as security, network management, and accessibility.
                    </p>
                  </div>
                  <div className="bg-purple-500/20 p-4 rounded-lg border border-purple-500/30">
                    <h3 className="text-lg font-bold text-white mb-2">Performance Cookies</h3>
                    <p className="text-gray-300 text-sm">
                      These cookies collect information about how visitors use our website, such as which pages are 
                      visited most often. This data helps us improve our website's performance.
                    </p>
                  </div>
                  <div className="bg-green-500/20 p-4 rounded-lg border border-green-500/30">
                    <h3 className="text-lg font-bold text-white mb-2">Functional Cookies</h3>
                    <p className="text-gray-300 text-sm">
                      These cookies allow the website to remember choices you make and provide enhanced, 
                      more personal features such as remembering your login details.
                    </p>
                  </div>
                  <div className="bg-orange-500/20 p-4 rounded-lg border border-orange-500/30">
                    <h3 className="text-lg font-bold text-white mb-2">Analytics Cookies</h3>
                    <p className="text-gray-300 text-sm">
                      We use analytics cookies to understand how our website is being used and to improve 
                      the user experience. These cookies collect anonymous information.
                    </p>
                  </div>
                </div>
              </div>

              {/* How We Use Cookies */}
              <div className="bg-white/5 rounded-lg p-8 border border-white/10">
                <div className="flex items-center mb-6">
                  <Database className="w-6 h-6 text-orange-400 mr-3" />
                  <h2 className="text-2xl font-bold text-white">How We Use Cookies</h2>
                </div>
                <div className="space-y-4 text-gray-300">
                  <p>PatternSight uses cookies for the following purposes:</p>
                  <ul className="list-disc list-inside space-y-2 ml-4">
                    <li>To authenticate users and prevent fraudulent use of user accounts</li>
                    <li>To remember your preferences and settings</li>
                    <li>To analyze site traffic and optimize our website performance</li>
                    <li>To provide personalized content and recommendations</li>
                    <li>To enable social media features and functionality</li>
                    <li>To deliver relevant advertisements (if applicable)</li>
                    <li>To conduct research and diagnostics to improve our services</li>
                  </ul>
                </div>
              </div>

              {/* Third-Party Cookies */}
              <div className="bg-white/5 rounded-lg p-8 border border-white/10">
                <div className="flex items-center mb-6">
                  <Eye className="w-6 h-6 text-orange-400 mr-3" />
                  <h2 className="text-2xl font-bold text-white">Third-Party Cookies</h2>
                </div>
                <div className="space-y-4 text-gray-300">
                  <p>
                    Some cookies on our site are placed by third-party services. We use the following third-party services 
                    that may place cookies on your device:
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-500/20 p-4 rounded-lg border border-gray-500/30">
                      <h4 className="font-bold text-white mb-2">Google Analytics</h4>
                      <p className="text-sm">For website analytics and performance monitoring</p>
                    </div>
                    <div className="bg-gray-500/20 p-4 rounded-lg border border-gray-500/30">
                      <h4 className="font-bold text-white mb-2">Supabase</h4>
                      <p className="text-sm">For authentication and database services</p>
                    </div>
                    <div className="bg-gray-500/20 p-4 rounded-lg border border-gray-500/30">
                      <h4 className="font-bold text-white mb-2">Vercel</h4>
                      <p className="text-sm">For website hosting and performance optimization</p>
                    </div>
                    <div className="bg-gray-500/20 p-4 rounded-lg border border-gray-500/30">
                      <h4 className="font-bold text-white mb-2">Social Media</h4>
                      <p className="text-sm">For social sharing and integration features</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Managing Cookies */}
              <div className="bg-white/5 rounded-lg p-8 border border-white/10">
                <div className="flex items-center mb-6">
                  <Shield className="w-6 h-6 text-orange-400 mr-3" />
                  <h2 className="text-2xl font-bold text-white">Managing Your Cookie Preferences</h2>
                </div>
                <div className="space-y-4 text-gray-300">
                  <p>
                    You have the right to decide whether to accept or reject cookies. You can exercise your cookie rights 
                    by setting your preferences in your browser settings.
                  </p>
                  <div className="bg-orange-500/20 p-4 rounded-lg border border-orange-500/30">
                    <h4 className="font-bold text-white mb-2">Browser Settings</h4>
                    <p className="text-sm">
                      Most web browsers allow you to control cookies through their settings preferences. 
                      However, limiting cookies may impact your experience on our website.
                    </p>
                  </div>
                  <div className="space-y-2">
                    <p className="font-semibold text-white">Popular browser cookie settings:</p>
                    <ul className="list-disc list-inside space-y-1 ml-4 text-sm">
                      <li><strong>Chrome:</strong> Settings → Privacy and security → Cookies and other site data</li>
                      <li><strong>Firefox:</strong> Options → Privacy & Security → Cookies and Site Data</li>
                      <li><strong>Safari:</strong> Preferences → Privacy → Cookies and website data</li>
                      <li><strong>Edge:</strong> Settings → Cookies and site permissions → Cookies and site data</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Cookie Retention */}
              <div className="bg-white/5 rounded-lg p-8 border border-white/10">
                <div className="flex items-center mb-6">
                  <Clock className="w-6 h-6 text-orange-400 mr-3" />
                  <h2 className="text-2xl font-bold text-white">Cookie Retention</h2>
                </div>
                <div className="space-y-4 text-gray-300">
                  <p>The length of time cookies remain on your device depends on their type:</p>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-blue-500/20 p-4 rounded-lg border border-blue-500/30">
                      <h4 className="font-bold text-white mb-2">Session Cookies</h4>
                      <p className="text-sm">Deleted when you close your browser</p>
                    </div>
                    <div className="bg-purple-500/20 p-4 rounded-lg border border-purple-500/30">
                      <h4 className="font-bold text-white mb-2">Persistent Cookies</h4>
                      <p className="text-sm">Remain until their expiration date or manual deletion</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Contact Information */}
              <div className="bg-gradient-to-r from-orange-500/20 to-pink-500/20 rounded-lg p-8 border border-orange-500/30">
                <h2 className="text-2xl font-bold text-white mb-4">Questions About Our Cookie Policy?</h2>
                <p className="text-gray-300 mb-4">
                  If you have any questions about our use of cookies or this Cookie Policy, please contact us:
                </p>
                <div className="space-y-2 text-gray-300">
                  <p><strong>Email:</strong> privacy@patternsight.com</p>
                </div>
                <div className="mt-6">
                  <motion.a
                    href="/contact"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-orange-500 to-pink-500 text-white font-semibold rounded-lg hover:from-orange-600 hover:to-pink-600 transition-all duration-200"
                  >
                    Contact Support
                  </motion.a>
                </div>
              </div>

              {/* Updates */}
              <div className="bg-white/5 rounded-lg p-8 border border-white/10">
                <h2 className="text-2xl font-bold text-white mb-4">Changes to This Cookie Policy</h2>
                <div className="space-y-4 text-gray-300">
                  <p>
                    We may update this Cookie Policy from time to time to reflect changes in our practices or for other 
                    operational, legal, or regulatory reasons.
                  </p>
                  <p>
                    When we make changes, we will update the "Last updated" date at the top of this policy and notify 
                    users through our website or other appropriate means.
                  </p>
                  <p>
                    We encourage you to review this Cookie Policy periodically to stay informed about our use of cookies.
                  </p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}

