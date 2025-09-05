'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, Shield, Clock, CheckCircle } from 'lucide-react';

interface DisclaimerModalProps {
  isOpen: boolean;
  onAccept: (initials: string) => void;
  onCancel: () => void;
}

export default function DisclaimerModal({ isOpen, onAccept, onCancel }: DisclaimerModalProps) {
  const [hasScrolledToEnd, setHasScrolledToEnd] = useState(false);
  const [confirmationText, setConfirmationText] = useState('');
  const [userInitials, setUserInitials] = useState('');
  const [hasAcceptedTerms, setHasAcceptedTerms] = useState(false);
  const [currentTime, setCurrentTime] = useState('');

  const requiredText = "I have read and understood the complete disclaimer.";

  useEffect(() => {
    if (isOpen) {
      setCurrentTime(new Date().toLocaleString());
    }
  }, [isOpen]);

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
    const scrolledToBottom = scrollTop + clientHeight >= scrollHeight - 10;
    if (scrolledToBottom) {
      setHasScrolledToEnd(true);
    }
  };

  const isFormValid = () => {
    return (
      hasScrolledToEnd &&
      confirmationText.toLowerCase().trim() === requiredText.toLowerCase() &&
      userInitials.trim().length >= 2 &&
      hasAcceptedTerms
    );
  };

  const handleAccept = () => {
    if (isFormValid()) {
      onAccept(userInitials);
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200 bg-gradient-to-r from-red-50 to-orange-50">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-red-500 to-orange-500 rounded-xl flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">🔮 PatternSight – Important Disclaimer</h2>
                <p className="text-sm text-red-600 font-medium">⚠️ MANDATORY LEGAL NOTICE ⚠️</p>
              </div>
            </div>
            <button
              onClick={onCancel}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-500" />
            </button>
          </div>

          {/* Scrollable Content */}
          <div 
            className="flex-1 overflow-y-auto p-6 space-y-6"
            onScroll={handleScroll}
          >
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
              <p className="text-red-800 font-medium text-center">
                You must read this entire disclaimer and provide explicit consent before accessing our services.
              </p>
            </div>

            {/* Section 1 */}
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-gray-900 flex items-center">
                <span className="w-8 h-8 bg-red-100 text-red-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">1</span>
                NO GUARANTEED RESULTS
              </h3>
              <div className="ml-11 space-y-2 text-gray-700">
                <p>PatternSight is a statistical analysis and entertainment platform.</p>
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>We do NOT guarantee winning lottery numbers, jackpot victories, or any financial returns.</li>
                  <li>Lottery drawings are random events. Past outcomes have no influence on future results.</li>
                  <li>All predictions provided by PatternSight are based on mathematical models, historical data analysis, and algorithmic pattern recognition.</li>
                  <li>Use this system for entertainment purposes only.</li>
                </ul>
              </div>
            </div>

            {/* Section 2 */}
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-gray-900 flex items-center">
                <span className="w-8 h-8 bg-orange-100 text-orange-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">2</span>
                RESPONSIBLE GAMBLING
              </h3>
              <div className="ml-11 space-y-2 text-gray-700">
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Lottery participation should be limited to amounts you can afford to lose.</li>
                  <li>Gambling can become addictive and harmful. If you or someone you know struggles with gambling, please seek help immediately:</li>
                  <li className="ml-4 text-blue-600 font-medium">Call the National Problem Gambling Helpline: 1-800-522-4700</li>
                  <li className="ml-4 text-blue-600 font-medium">Visit ncpgambling.org for resources.</li>
                  <li>PatternSight encourages healthy, responsible gaming practices and does not endorse or promote excessive gambling.</li>
                </ul>
              </div>
            </div>

            {/* Section 3 */}
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-gray-900 flex items-center">
                <span className="w-8 h-8 bg-yellow-100 text-yellow-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">3</span>
                LIMITATIONS OF STATISTICAL ANALYSIS
              </h3>
              <div className="ml-11 space-y-2 text-gray-700">
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Our system uses advanced statistical methods, AI models, astronomical correlations, and numerological analysis to detect patterns in historical lottery data.</li>
                  <li>These insights are not predictions of certainty.</li>
                  <li>Every lottery combination has the same mathematical probability of being drawn, regardless of past trends.</li>
                </ul>
              </div>
            </div>

            {/* Section 4 */}
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-gray-900 flex items-center">
                <span className="w-8 h-8 bg-green-100 text-green-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">4</span>
                FINANCIAL DISCLAIMER
              </h3>
              <div className="ml-11 space-y-2 text-gray-700">
                <ul className="list-disc list-inside space-y-1 ml-4">
                  <li>Subscription fees are solely for access to our analysis platform and entertainment services.</li>
                  <li>These fees do not guarantee any lottery winnings or financial return.</li>
                  <li>You are solely responsible for:
                    <ul className="list-disc list-inside ml-6 mt-1">
                      <li>The amounts you choose to gamble.</li>
                      <li>Any financial consequences resulting from your decisions.</li>
                    </ul>
                  </li>
                </ul>
              </div>
            </div>

            {/* Sections 5-13 (abbreviated for space, but would include all) */}
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-gray-900 flex items-center">
                <span className="w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">5</span>
                ACCURACY AND RELIABILITY
              </h3>
              <div className="ml-11 space-y-2 text-gray-700">
                <p>While we aim to provide accurate, data-driven insights, absolute accuracy cannot be guaranteed.</p>
                <p>Always verify results through official lottery sources before acting on them.</p>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-bold text-gray-900 flex items-center">
                <span className="w-8 h-8 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">6</span>
                AGE & LEGAL REQUIREMENTS
              </h3>
              <div className="ml-11 space-y-2 text-gray-700">
                <p>You must be at least 18 years old, or meet the legal gambling age in your jurisdiction, to use PatternSight.</p>
                <p>You are responsible for ensuring compliance with all applicable laws in your region.</p>
              </div>
            </div>

            <div className="space-y-4">
              <h3 className="text-lg font-bold text-gray-900 flex items-center">
                <span className="w-8 h-8 bg-red-100 text-red-600 rounded-full flex items-center justify-center text-sm font-bold mr-3">7</span>
                LIMITATION OF LIABILITY
              </h3>
              <div className="ml-11 space-y-2 text-gray-700">
                <p>PatternSight, its owners, employees, and affiliates shall not be liable for financial losses, technical failures, or any damages.</p>
                <p className="font-medium text-red-600">By using PatternSight, you agree that all risks remain entirely your own.</p>
              </div>
            </div>

            {/* Final Warning */}
            <div className="bg-red-50 border-2 border-red-200 rounded-lg p-6 mt-8">
              <h3 className="text-xl font-bold text-red-800 mb-4 flex items-center">
                <AlertTriangle className="w-6 h-6 mr-2" />
                FINAL WARNING
              </h3>
              <div className="space-y-2 text-red-700">
                <p>By proceeding, you acknowledge that PatternSight is an entertainment service only.</p>
                <ul className="list-disc list-inside ml-4 space-y-1">
                  <li>We do not guarantee lottery wins or financial outcomes.</li>
                  <li>Lottery participation always involves financial risk.</li>
                  <li>Please gamble responsibly and within your means.</li>
                </ul>
              </div>
            </div>

            {/* Scroll indicator */}
            {!hasScrolledToEnd && (
              <div className="sticky bottom-0 bg-gradient-to-t from-white via-white to-transparent pt-8 pb-4">
                <div className="flex items-center justify-center text-orange-600">
                  <AlertTriangle className="w-5 h-5 mr-2" />
                  <span className="font-medium">Please scroll to the end of the disclaimer to continue</span>
                </div>
              </div>
            )}
          </div>

          {/* Consent Form */}
          {hasScrolledToEnd && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="border-t border-gray-200 p-6 bg-gray-50"
            >
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                <Shield className="w-5 h-5 mr-2 text-green-600" />
                CONSENT & ACKNOWLEDGMENT
              </h3>

              <div className="space-y-4">
                {/* Confirmation Text */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Type the following text to confirm you have read and understood this disclaimer:
                  </label>
                  <p className="text-sm text-gray-600 mb-2 italic">"{requiredText}"</p>
                  <input
                    type="text"
                    value={confirmationText}
                    onChange={(e) => setConfirmationText(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                    placeholder="Type the confirmation text exactly as shown above"
                  />
                  {confirmationText && (
                    <div className="mt-1 flex items-center">
                      {confirmationText.toLowerCase().trim() === requiredText.toLowerCase() ? (
                        <CheckCircle className="w-4 h-4 text-green-500 mr-1" />
                      ) : (
                        <X className="w-4 h-4 text-red-500 mr-1" />
                      )}
                      <span className={`text-sm ${
                        confirmationText.toLowerCase().trim() === requiredText.toLowerCase() 
                          ? 'text-green-600' 
                          : 'text-red-600'
                      }`}>
                        {confirmationText.toLowerCase().trim() === requiredText.toLowerCase() 
                          ? 'Text matches correctly' 
                          : 'Text does not match'}
                      </span>
                    </div>
                  )}
                </div>

                {/* User Initials */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Enter your initials for legal acknowledgment:
                  </label>
                  <input
                    type="text"
                    value={userInitials}
                    onChange={(e) => setUserInitials(e.target.value.toUpperCase())}
                    className="w-32 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-orange-500"
                    placeholder="J.D."
                    maxLength={10}
                  />
                </div>

                {/* Final Checkbox */}
                <div className="flex items-start space-x-3">
                  <input
                    type="checkbox"
                    id="final-consent"
                    checked={hasAcceptedTerms}
                    onChange={(e) => setHasAcceptedTerms(e.target.checked)}
                    className="mt-1 w-4 h-4 text-orange-600 border-gray-300 rounded focus:ring-orange-500"
                  />
                  <label htmlFor="final-consent" className="text-sm text-gray-700">
                    ☑ I confirm that I have read, understood, and agree to all terms in this disclaimer. 
                    I acknowledge that PatternSight provides entertainment services only and does not guarantee lottery winnings.
                  </label>
                </div>

                {/* Timestamp */}
                <div className="text-xs text-gray-500 flex items-center space-x-4">
                  <span className="flex items-center">
                    <Clock className="w-3 h-3 mr-1" />
                    Timestamp: {currentTime}
                  </span>
                  <span>IP: [Auto-logged for legal compliance]</span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3 mt-6">
                <button
                  onClick={onCancel}
                  className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleAccept}
                  disabled={!isFormValid()}
                  className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
                    isFormValid()
                      ? 'bg-gradient-to-r from-orange-500 to-pink-500 text-white hover:from-orange-600 hover:to-pink-600'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  Accept & Continue
                </button>
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </AnimatePresence>
  );
}

