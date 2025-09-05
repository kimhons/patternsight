'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

interface User {
  id: string;
  email: string;
  full_name: string;
  created_at: string;
  updated_at: string;
  subscription_tier?: 'starter' | 'pro' | 'elite';
  subscription_status?: 'active' | 'inactive' | 'cancelled' | 'trial';
  subscription_end_date?: string;
  daily_analysis_limit: number;
  daily_analysis_used: number;
  total_analyses: number;
  avatar_url?: string;
  company?: string;
  role?: string;
}

interface AuthContextType {
  user: User | null;
  supabaseUser: any | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<{ error?: any }>;
  signUp: (email: string, password: string, fullName: string) => Promise<{ error?: any }>;
  signOut: () => Promise<{ error?: any }>;
  updateProfile: (updates: Partial<User>) => Promise<{ error?: any }>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Mock user for development
    const mockUser: User = {
      id: '1',
      email: 'demo@patternsight.com',
      full_name: 'Pattern Analyst',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      subscription_tier: 'pro',
      subscription_status: 'active',
      daily_analysis_limit: 100,
      daily_analysis_used: 23,
      total_analyses: 1247,
      company: 'PatternSight Demo',
      role: 'Data Scientist'
    };

    setTimeout(() => {
      setUser(mockUser);
      setLoading(false);
    }, 1000);
  }, []);

  const handleSignIn = async (email: string, password: string) => {
    setLoading(true);
    try {
      // Mock sign in
      await new Promise(resolve => setTimeout(resolve, 1000));
      return { error: null };
    } catch (error) {
      return { error };
    } finally {
      setLoading(false);
    }
  };

  const handleSignUp = async (email: string, password: string, fullName: string) => {
    setLoading(true);
    try {
      // Mock sign up
      await new Promise(resolve => setTimeout(resolve, 1000));
      return { error: null };
    } catch (error) {
      return { error };
    } finally {
      setLoading(false);
    }
  };

  const handleSignOut = async () => {
    setLoading(true);
    try {
      setUser(null);
      return { error: null };
    } catch (error) {
      return { error };
    } finally {
      setLoading(false);
    }
  };

  const updateProfile = async (updates: Partial<User>) => {
    if (!user) return { error: 'No user logged in' };

    try {
      setUser({ ...user, ...updates });
      return { error: null };
    } catch (error) {
      return { error };
    }
  };

  const value: AuthContextType = {
    user,
    supabaseUser: null,
    loading,
    signIn: handleSignIn,
    signUp: handleSignUp,
    signOut: handleSignOut,
    updateProfile
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

