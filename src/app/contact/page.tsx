'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Footer from '@/components/layout/Footer';
import { Mail, Phone, MessageCircle, Clock, Shield, AlertTriangle, Users, Building, Lock, HelpCircle, ChevronRight } from 'lucide-react';

export default function Contact() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    category: '',
    priority: 'normal',
    subject: '',
    message: ''
  });

  const contactMethods = [
    {
      icon: <Mail className="w-6 h-6" />,
      title: "Email Support",
      email: "support@patternsight.com",
      description: "Response within 24 hours",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <AlertTriangle className="w-6 h-6" />,
      title: "Urgent Issues",
      email: "urgent@patternsight.com",
      description: "For billing or security emergencies",
      color: "from-red-500 to-orange-500"
    },
    {
      icon: <Building className="w-6 h-6" />,
      title: "Business Inquiries",
      email: "business@patternsight.com",
      description: "Partnerships and enterprise solutions",
      color: "from-purple-500 to-indigo-500"
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "Legal Matters",
      email: "legal@patternsight.com",
      description: "Terms, privacy, and compliance",
      color: "from-green-500 to-emerald-500"
    },
    {
      icon: <Lock className="w-6 h-6" />,
      title: "Privacy Concerns",
      email: "privacy@patternsight.com",
      description: "Data protection and privacy requests",
      color: "from-pink-500 to-rose-500"
    }
  ];

  const otherWays = [
    {
      icon: <MessageCircle className="w-8 h-8" />,
      title: "Live Chat",
      description: "Chat with our support team in real-time",
      availability: "Available during business hours",
      action: "Start Chat"
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: "Social Media",
      description: "Follow us for updates and quick support",
      availability: "Twitter @PatternSight, Facebook /PatternSight, Instagram @PatternSight",
      action: "Follow Us"
    }
  ];

  const faqs = [
    {
      question: "How quickly will I get a response?",
      answer: "General inquiries: 24 hours. Technical issues: 12 hours. Urgent matters: 4 hours. We prioritize responses based on issue severity and customer needs."
    },
    {
      question: "What information should I include?",
      answer: "Include your account email, describe the issue clearly, mention what you've tried, and provide any error messages. The more details you provide, the faster we can help."
    },
    {
      question: "Can I schedule a call?",
      answer: "For complex issues or business inquiries, we can arrange phone consultations. Contact us via email to schedule a call during business hours."
    },
    {
      question: "Do you offer phone support?",
      answer: "Currently, we provide support primarily through email and live chat. Phone support is available for urgent billing issues and enterprise customers."
    },
    {
      question: "What languages do you support?",
      answer: "Our primary support language is English. We're working to expand language support and can provide basic assistance in Spanish and French."
    },
    {
      question: "How do I report a bug or issue?",
      answer: "Use the contact form above with \"Technical Issues\" category. Include detailed steps to reproduce the problem, your browser/device info, and any error messages."
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
                <p className="text-xl text-orange-400">Contact Us</p>
              </div>
            </div>
            <p className="text-2xl text-gray-300">
              We're here to help with any questions or concerns
            </p>
          </motion.div>
        </div>
      </section>

      {/* Get in Touch Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-white mb-6">📞 Get in Touch</h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
            {contactMethods.map((method, index) => (
              <motion.div
                key={method.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-6 border border-white/10 hover:bg-white/10 transition-all duration-200"
              >
                <div className={`w-12 h-12 bg-gradient-to-r ${method.color} rounded-lg flex items-center justify-center mb-4 text-white`}>
                  {method.icon}
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{method.title}</h3>
                <a href={`mailto:${method.email}`} className="text-orange-400 hover:text-orange-300 transition-colors duration-200 block mb-2">
                  {method.email}
                </a>
                <p className="text-gray-300 text-sm">{method.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Business Hours & Response Times */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 max-w-6xl mx-auto">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              className="bg-white/5 rounded-lg p-8 border border-white/10"
            >
              <div className="flex items-center mb-6">
                <Clock className="w-6 h-6 text-orange-400 mr-3" />
                <h3 className="text-2xl font-bold text-white">🕒 Business Hours</h3>
              </div>
              <div className="space-y-3 text-gray-300">
                <div className="flex justify-between">
                  <span className="font-semibold">Monday - Friday:</span>
                  <span>9:00 AM - 9:00 PM EST</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-semibold">Saturday:</span>
                  <span>10:00 AM - 6:00 PM EST</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-semibold">Sunday:</span>
                  <span>12:00 PM - 6:00 PM EST</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-semibold">Holidays:</span>
                  <span>Limited hours</span>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              className="bg-white/5 rounded-lg p-8 border border-white/10"
            >
              <div className="flex items-center mb-6">
                <Shield className="w-6 h-6 text-orange-400 mr-3" />
                <h3 className="text-2xl font-bold text-white">⏱️ Response Times</h3>
              </div>
              <div className="space-y-4">
                <div className="bg-green-500/20 p-3 rounded-lg border border-green-500/30">
                  <div className="flex justify-between items-center">
                    <span className="text-white font-semibold">General Support</span>
                    <span className="text-green-300">Within 24 hours</span>
                  </div>
                </div>
                <div className="bg-blue-500/20 p-3 rounded-lg border border-blue-500/30">
                  <div className="flex justify-between items-center">
                    <span className="text-white font-semibold">Technical Issues</span>
                    <span className="text-blue-300">Within 12 hours</span>
                  </div>
                </div>
                <div className="bg-orange-500/20 p-3 rounded-lg border border-orange-500/30">
                  <div className="flex justify-between items-center">
                    <span className="text-white font-semibold">Urgent/Billing</span>
                    <span className="text-orange-300">Within 4 hours</span>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Contact Form */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center mb-12"
            >
              <h2 className="text-4xl font-bold text-white mb-6">📝 Send us a Message</h2>
            </motion.div>

            <motion.form
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="bg-white/5 rounded-lg p-8 border border-white/10"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-white font-semibold mb-2">Full Name *</label>
                  <input
                    type="text"
                    placeholder="Your full name"
                    value={formData.name}
                    onChange={(e) => setFormData({...formData, name: e.target.value})}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  />
                </div>
                <div>
                  <label className="block text-white font-semibold mb-2">Email Address *</label>
                  <input
                    type="email"
                    placeholder="your.email@example.com"
                    value={formData.email}
                    onChange={(e) => setFormData({...formData, email: e.target.value})}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-white font-semibold mb-2">Category *</label>
                  <select 
                    value={formData.category}
                    onChange={(e) => setFormData({...formData, category: e.target.value})}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  >
                    <option value="">General Support</option>
                    <option value="technical">Technical Issues</option>
                    <option value="billing">Billing Questions</option>
                    <option value="feature">Feature Request</option>
                  </select>
                </div>
                <div>
                  <label className="block text-white font-semibold mb-2">Priority</label>
                  <select 
                    value={formData.priority}
                    onChange={(e) => setFormData({...formData, priority: e.target.value})}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                  >
                    <option value="normal">Normal - Standard support</option>
                    <option value="high">High - Urgent issue</option>
                    <option value="low">Low - General inquiry</option>
                  </select>
                </div>
              </div>

              <div className="mb-6">
                <label className="block text-white font-semibold mb-2">Subject *</label>
                <input
                  type="text"
                  placeholder="Brief description of your inquiry"
                  value={formData.subject}
                  onChange={(e) => setFormData({...formData, subject: e.target.value})}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent"
                />
              </div>

              <div className="mb-6">
                <label className="block text-white font-semibold mb-2">Message *</label>
                <textarea
                  rows={6}
                  placeholder="Please provide as much detail as possible about your question or issue. Include any error messages, steps you've tried, and relevant account information."
                  value={formData.message}
                  onChange={(e) => setFormData({...formData, message: e.target.value})}
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none"
                ></textarea>
              </div>

              <div className="mb-8">
                <div className="bg-blue-500/20 p-4 rounded-lg border border-blue-500/30">
                  <h4 className="text-white font-semibold mb-2">💡 For Faster Support</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• Include your account email address</li>
                    <li>• Describe steps you've already tried</li>
                    <li>• Mention your browser and device type</li>
                    <li>• Include any error messages you've seen</li>
                    <li>• Specify which features you're having trouble with</li>
                  </ul>
                </div>
              </div>

              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="w-full bg-gradient-to-r from-orange-500 to-pink-500 text-white font-semibold py-4 px-8 rounded-lg hover:from-orange-600 hover:to-pink-600 transition-all duration-200"
              >
                Send Message 📤
              </motion.button>
            </motion.form>
          </div>
        </div>
      </section>

      {/* Other Ways to Reach Us */}
      <section className="py-16 bg-black/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-white mb-6">🌐 Other Ways to Reach Us</h2>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {otherWays.map((method, index) => (
              <motion.div
                key={method.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg p-8 border border-white/10 text-center hover:bg-white/10 transition-all duration-200"
              >
                <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-pink-500 rounded-lg flex items-center justify-center mx-auto mb-4 text-white">
                  {method.icon}
                </div>
                <h3 className="text-xl font-bold text-white mb-2">{method.title}</h3>
                <p className="text-gray-300 mb-3">{method.description}</p>
                <p className="text-gray-400 text-sm mb-4">{method.availability}</p>
                <button className="bg-gradient-to-r from-orange-500 to-pink-500 text-white px-6 py-2 rounded-lg hover:from-orange-600 hover:to-pink-600 transition-all duration-200">
                  {method.action}
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-12"
          >
            <h2 className="text-4xl font-bold text-white mb-6">❓ Frequently Asked Questions</h2>
          </motion.div>

          <div className="max-w-4xl mx-auto space-y-6">
            {faqs.map((faq, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="bg-white/5 rounded-lg border border-white/10"
              >
                <div className="p-6">
                  <h3 className="text-lg font-bold text-white mb-3 flex items-center">
                    <HelpCircle className="w-5 h-5 text-orange-400 mr-2" />
                    {faq.question}
                  </h3>
                  <p className="text-gray-300 leading-relaxed">{faq.answer}</p>
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

