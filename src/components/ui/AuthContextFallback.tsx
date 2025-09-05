'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';

interface User {
  id: string;
  email: string;
  name: string;
  plan: 'starter' | 'pro' | 'elite';
  predictionsUsed: number;
  predictionsLimit: number;
  disclaimerAccepted: boolean;
  disclaimerTimestamp?: string;
  userInitials?: string;
  subscriptionStatus: 'active' | 'inactive' | 'trial' | 'cancelled' | 'past_due';
  createdAt: string;
  lastLogin: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  updateUser: (userData: Partial<User>) => void;
  isAuthenticated: boolean;
  checkAuth: () => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  // Generate consistent user ID based on email
  const generateUserId = (email: string): string => {
    return `user_${btoa(email).replace(/[^a-zA-Z0-9]/g, '').substring(0, 12)}`;
  };

  // Check authentication status on mount
  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = (): boolean => {
    try {
      const userData = localStorage.getItem('patternsight_user');
      const authToken = localStorage.getItem('patternsight_auth_token');
      
      if (userData && authToken) {
        const parsedUser = JSON.parse(userData);
        
        // Ensure user has required fields
        if (parsedUser.email) {
          // Generate ID if missing
          if (!parsedUser.id) {
            parsedUser.id = generateUserId(parsedUser.email);
          }
          
          // Set default values if missing
          const completeUser: User = {
            id: parsedUser.id,
            email: parsedUser.email,
            name: parsedUser.name || parsedUser.email.split('@')[0],
            plan: parsedUser.plan || 'starter',
            predictionsUsed: parsedUser.predictionsUsed || 0,
            predictionsLimit: parsedUser.predictionsLimit || getPlanLimit(parsedUser.plan || 'starter'),
            disclaimerAccepted: parsedUser.disclaimerAccepted || false,
            disclaimerTimestamp: parsedUser.disclaimerTimestamp,
            userInitials: parsedUser.userInitials,
            subscriptionStatus: parsedUser.subscriptionStatus || 'trial',
            createdAt: parsedUser.createdAt || new Date().toISOString(),
            lastLogin: new Date().toISOString()
          };
          
          setUser(completeUser);
          
          // Update localStorage with complete user data
          localStorage.setItem('patternsight_user', JSON.stringify(completeUser));
          
          setLoading(false);
          return true;
        }
      }
      
      setUser(null);
      setLoading(false);
      return false;
    } catch (error) {
      console.error('Auth check error:', error);
      setUser(null);
      setLoading(false);
      return false;
    }
  };

  const getPlanLimit = (plan: string): number => {
    switch (plan) {
      case 'starter': return 3;
      case 'pro': return 25;
      case 'elite': return 50;
      default: return 3;
    }
  };

  const login = async (email: string, password: string): Promise<boolean> => {
    setLoading(true);
    
    try {
      // Simulate authentication API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // For demo purposes, accept any email/password combination
      const userId = generateUserId(email);
      
      // Check if user already exists
      const existingUserData = localStorage.getItem('patternsight_user');
      let userData: User;
      
      if (existingUserData) {
        const existingUser = JSON.parse(existingUserData);
        if (existingUser.email === email) {
          // Returning user
          userData = {
            ...existingUser,
            lastLogin: new Date().toISOString()
          };
        } else {
          // New user with different email
          userData = {
            id: userId,
            email,
            name: email.split('@')[0],
            plan: 'starter',
            predictionsUsed: 0,
            predictionsLimit: 3,
            disclaimerAccepted: false,
            subscriptionStatus: 'trial',
            createdAt: new Date().toISOString(),
            lastLogin: new Date().toISOString()
          };
        }
      } else {
        // Completely new user
        userData = {
          id: userId,
          email,
          name: email.split('@')[0],
          plan: 'starter',
          predictionsUsed: 0,
          predictionsLimit: 3,
          disclaimerAccepted: false,
          subscriptionStatus: 'trial',
          createdAt: new Date().toISOString(),
          lastLogin: new Date().toISOString()
        };
      }
      
      // Store auth token and user data
      const authToken = `token_${userId}_${Date.now()}`;
      localStorage.setItem('patternsight_auth_token', authToken);
      localStorage.setItem('patternsight_user', JSON.stringify(userData));
      
      setUser(userData);
      setLoading(false);
      return true;
      
    } catch (error) {
      console.error('Login error:', error);
      setLoading(false);
      return false;
    }
  };

  const logout = () => {
    localStorage.removeItem('patternsight_user');
    localStorage.removeItem('patternsight_auth_token');
    
    // Clear usage data
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('usage_') || key.startsWith('bundles_')) {
        localStorage.removeItem(key);
      }
    });
    
    setUser(null);
    router.push('/auth/signin');
  };

  const updateUser = (userData: Partial<User>) => {
    if (!user) return;
    
    const updatedUser = { ...user, ...userData };
    setUser(updatedUser);
    localStorage.setItem('patternsight_user', JSON.stringify(updatedUser));
  };

  const isAuthenticated = user !== null && user.disclaimerAccepted;

  const value: AuthContextType = {
    user,
    loading,
    login,
    logout,
    updateUser,
    isAuthenticated,
    checkAuth
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

