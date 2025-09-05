'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs';
import type { User, Session } from '@supabase/supabase-js';

export interface UserProfile {
  id: string;
  email: string;
  full_name?: string;
  subscription_tier: 'free' | 'starter' | 'pro' | 'elite';
  predictions_remaining: number;
  total_predictions: number;
  created_at: string;
  subscription_expires?: string;
  daily_limit: number;
  daily_used: number;
  last_reset_date: string;
  disclaimer_accepted: boolean;
  disclaimer_accepted_at?: string;
  disclaimer_version?: string;
  disclaimer_initials?: string;
}

interface AuthContextType {
  user: User | null;
  session: Session | null;
  profile: UserProfile | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<{ error?: any }>;
  signUp: (email: string, password: string, fullName?: string) => Promise<{ error?: any }>;
  signInWithGoogle: () => Promise<{ error?: any }>;
  signInWithApple: () => Promise<{ error?: any }>;
  signOut: () => Promise<void>;
  updateProfile: (updates: Partial<UserProfile>) => Promise<void>;
  acceptDisclaimer: (initials: string) => Promise<void>;
  canGeneratePrediction: () => boolean;
  decrementPredictions: () => Promise<void>;
  upgradeTier: (tier: 'starter' | 'pro' | 'elite') => Promise<void>;
  getTierLimits: (tier: string) => { daily: number; description: string };
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  
  const supabase = createClientComponentClient();

  const getTierLimits = (tier: string) => {
    const limits = {
      free: { daily: 1, description: '1 prediction per day' },
      starter: { daily: 3, description: '3 predictions per day' },
      pro: { daily: 25, description: '25 predictions per day' },
      elite: { daily: 50, description: '50 predictions per day' }
    };
    return limits[tier as keyof typeof limits] || limits.free;
  };

  useEffect(() => {
    // Get initial session
    const getInitialSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setSession(session);
      setUser(session?.user ?? null);
      
      if (session?.user) {
        await loadUserProfile(session.user.id);
      }
      
      setLoading(false);
    };

    getInitialSession();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        setSession(session);
        setUser(session?.user ?? null);
        
        if (session?.user) {
          await loadUserProfile(session.user.id);
        } else {
          setProfile(null);
        }
        
        setLoading(false);
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  const loadUserProfile = async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('*')
        .eq('id', userId)
        .single();

      if (error && error.code === 'PGRST116') {
        // Profile doesn't exist, create one
        await createUserProfile(userId);
        return;
      }

      if (error) {
        console.error('Error loading profile:', error);
        return;
      }

      // Check if daily limit needs reset
      const today = new Date().toISOString().split('T')[0];
      if (data.last_reset_date !== today) {
        const tierLimits = getTierLimits(data.subscription_tier);
        await updateProfile({
          daily_used: 0,
          last_reset_date: today,
          predictions_remaining: tierLimits.daily === -1 ? 999999 : tierLimits.daily
        });
      } else {
        setProfile(data);
      }
    } catch (error) {
      console.error('Error in loadUserProfile:', error);
    }
  };

  const createUserProfile = async (userId: string) => {
    try {
      const today = new Date().toISOString().split('T')[0];
      const newProfile: Partial<UserProfile> = {
        id: userId,
        email: user?.email || '',
        full_name: user?.user_metadata?.full_name || 'User',
        subscription_tier: 'free',
        predictions_remaining: 3,
        total_predictions: 0,
        created_at: new Date().toISOString(),
        daily_limit: 3,
        daily_used: 0,
        last_reset_date: today,
        disclaimer_accepted: false,
        disclaimer_version: '1.0'
      };

      const { data, error } = await supabase
        .from('profiles')
        .insert([newProfile])
        .select()
        .single();

      if (error) {
        console.error('Error creating profile:', error);
        return;
      }

      setProfile(data);
    } catch (error) {
      console.error('Error in createUserProfile:', error);
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      const { error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      return { error };
    } catch (error) {
      return { error };
    }
  };

  const signUp = async (email: string, password: string, fullName?: string) => {
    try {
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: fullName,
          },
        },
      });

      return { error };
    } catch (error) {
      return { error };
    }
  };

  const signInWithGoogle = async () => {
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
          redirectTo: `${window.location.origin}/dashboard`,
          queryParams: {
            access_type: 'offline',
            prompt: 'consent',
          },
        },
      });

      return { error };
    } catch (error) {
      return { error };
    }
  };

  const signInWithApple = async () => {
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'apple',
        options: {
          redirectTo: `${window.location.origin}/dashboard`,
        },
      });

      return { error };
    } catch (error) {
      return { error };
    }
  };

  const signOut = async () => {
    await supabase.auth.signOut();
  };

  const updateProfile = async (updates: Partial<UserProfile>) => {
    if (!user || !profile) return;

    try {
      const { data, error } = await supabase
        .from('profiles')
        .update(updates)
        .eq('id', user.id)
        .select()
        .single();

      if (error) {
        console.error('Error updating profile:', error);
        return;
      }

      setProfile(data);
    } catch (error) {
      console.error('Error in updateProfile:', error);
    }
  };

  const canGeneratePrediction = () => {
    if (!profile) return false;
    
    // Elite tier has unlimited predictions
    if (profile.subscription_tier === 'elite') return true;
    
    // Other tiers check remaining predictions
    return profile.predictions_remaining > 0;
  };

  const decrementPredictions = async () => {
    if (!profile || !user) return;
    
    // Elite tier doesn't decrement
    if (profile.subscription_tier === 'elite') return;
    
    const newRemaining = Math.max(0, profile.predictions_remaining - 1);
    const newTotal = profile.total_predictions + 1;
    const newDailyUsed = profile.daily_used + 1;
    
    await updateProfile({
      predictions_remaining: newRemaining,
      total_predictions: newTotal,
      daily_used: newDailyUsed
    });
  };

  const acceptDisclaimer = async (initials: string) => {
    if (!profile) return;
    
    await updateProfile({
      disclaimer_accepted: true,
      disclaimer_accepted_at: new Date().toISOString(),
      disclaimer_version: '1.0',
      disclaimer_initials: initials
    });
  };

  const upgradeTier = async (tier: 'starter' | 'pro' | 'elite') => {
    if (!profile) return;
    
    const tierLimits = getTierLimits(tier);
    const today = new Date().toISOString().split('T')[0];
    
    await updateProfile({
      subscription_tier: tier,
      daily_limit: tierLimits.daily === -1 ? 999999 : tierLimits.daily,
      predictions_remaining: tierLimits.daily === -1 ? 999999 : tierLimits.daily,
      daily_used: 0,
      last_reset_date: today,
      subscription_expires: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString() // 30 days
    });
  };

  const value: AuthContextType = {
    user,
    session,
    profile,
    loading,
    signIn,
    signUp,
    signInWithGoogle,
    signInWithApple,
    signOut,
    updateProfile,
    acceptDisclaimer,
    canGeneratePrediction,
    decrementPredictions,
    upgradeTier,
    getTierLimits,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

