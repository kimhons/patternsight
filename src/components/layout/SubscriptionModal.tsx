'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '@/components/ui/AuthContextFixed';

interface SubscriptionModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SubscriptionModal({ isOpen, onClose }: SubscriptionModalProps) {
  const [selectedTier, setSelectedTier] = useState<'starter' | 'pro' | 'elite'>('starter');
  const [loading, setLoading] = useState(false);
  const { profile, upgradeTier, getTierLimits } = useAuth();

  const tiers = [
    {
      id: 'starter' as const,
      name: 'Oracle Starter',
      price: '$3.99',
      predictions: '3 per day',
      features: [
        '✓ Basic pattern analysis',
        '✓ Daily cosmic insights',
        '✓ Community access',
        '✓ Enhanced UPPS v3.0 access'
      ],
      color: 'from-blue-500 to-cyan-500'
    },
    {
      id: 'pro' as const,
      name: 'Cosmic Oracle Pro',
      price: '$19.99',
      predictions: '25 per day',
      features: [
        '✓ Advanced AI analysis',
        '✓ Cosmic intelligence',
        '✓ Priority support',
        '✓ All 5 academic pillars',
        '✓ Detailed explanations'
      ],
      color: 'from-orange-500 to-pink-500',
      popular: true
    },
    {
      id: 'elite' as const,
      name: 'Cosmic Oracle Elite',
      price: '$49.99',
      predictions: '50 per day',
      features: [
        '✓ Maximum AI power',
        '✓ Personal cosmic profile',
        '✓ VIP support',
        '✓ API access',
        '✓ White-label options',
        '✓ Advanced analytics'
      ],
      color: 'from-purple-500 to-indigo-500'
    }
  ];

  const handleUpgrade = async () => {
    setLoading(true);
    try {
      await upgradeTier(selectedTier);
      onClose();
      // In a real app, you'd integrate with Stripe or another payment processor here
      alert(`Successfully upgraded to ${selectedTier} tier! (Demo mode - no payment processed)`);
    } catch (error) {
      console.error('Error upgrading tier:', error);
      alert('Error upgrading subscription. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 rounded-lg p-8 w-full max-w-4xl border border-white/20 max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-white mb-2">
              Upgrade Your Intelligence Level
            </h2>
            <p className="text-gray-300">
              Choose your subscription tier to unlock Enhanced UPPS v3.0 features
            </p>
            {profile && (
              <div className="mt-4 p-3 bg-white/10 rounded-lg">
                <p className="text-sm text-gray-300">
                  Current Tier: <span className="font-bold text-orange-400 capitalize">{profile.subscription_tier}</span>
                  {' • '}
                  Predictions Remaining: <span className="font-bold text-green-400">{profile.predictions_remaining}</span>
                </p>
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {tiers.map((tier) => (
              <motion.div
                key={tier.id}
                className={`relative p-6 rounded-lg border-2 cursor-pointer transition-all duration-200 ${
                  selectedTier === tier.id
                    ? 'border-orange-500 bg-white/10'
                    : 'border-white/20 bg-white/5 hover:bg-white/10'
                } ${tier.popular ? 'ring-2 ring-orange-500' : ''}`}
                onClick={() => setSelectedTier(tier.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {tier.popular && (
                  <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                    <span className="bg-gradient-to-r from-orange-500 to-pink-500 text-white px-3 py-1 rounded-full text-xs font-bold">
                      MOST POPULAR
                    </span>
                  </div>
                )}

                <div className="text-center mb-4">
                  <h3 className="text-xl font-bold text-white mb-2">{tier.name}</h3>
                  <div className={`text-3xl font-bold bg-gradient-to-r ${tier.color} bg-clip-text text-transparent mb-1`}>
                    {tier.price}
                  </div>
                  <p className="text-gray-400 text-sm">{tier.predictions}</p>
                </div>

                <ul className="space-y-2 mb-6">
                  {tier.features.map((feature, index) => (
                    <li key={index} className="text-sm text-gray-300">
                      {feature}
                    </li>
                  ))}
                </ul>

                <div className="flex items-center justify-center">
                  <div className={`w-4 h-4 rounded-full border-2 ${
                    selectedTier === tier.id
                      ? 'border-orange-500 bg-orange-500'
                      : 'border-gray-400'
                  }`}>
                    {selectedTier === tier.id && (
                      <div className="w-2 h-2 bg-white rounded-full m-0.5"></div>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          <div className="flex justify-center space-x-4">
            <motion.button
              onClick={onClose}
              className="px-6 py-3 bg-gray-600 hover:bg-gray-700 text-white rounded-lg font-semibold transition-all duration-200"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Cancel
            </motion.button>
            
            <motion.button
              onClick={handleUpgrade}
              disabled={loading}
              className="px-8 py-3 bg-gradient-to-r from-orange-500 to-pink-500 hover:from-orange-600 hover:to-pink-600 disabled:opacity-50 text-white rounded-lg font-semibold transition-all duration-200"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {loading ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Upgrading...
                </div>
              ) : (
                `Upgrade to ${tiers.find(t => t.id === selectedTier)?.name}`
              )}
            </motion.button>
          </div>

          <button
            onClick={onClose}
            className="absolute top-4 right-4 text-gray-400 hover:text-white"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

