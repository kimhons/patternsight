import { createClient } from '@supabase/supabase-js'
import { createBrowserClient } from '@supabase/ssr'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://xasxypwfdosdkkwqvsru.supabase.co'
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhhc3h5cHdmZG9zZGtrd3F2c3J1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTY5Mjc5NjMsImV4cCI6MjA3MjUwMzk2M30.mE_6JjBIpjz_4UDW9QuJQ1PUmmcujl-zVtn2ybXW8_k'

// Client-side Supabase client
export const supabase = createBrowserClient(supabaseUrl, supabaseAnonKey)

// Server-side Supabase client (for API routes)
export const supabaseAdmin = createClient(
  supabaseUrl,
  process.env.SUPABASE_SERVICE_ROLE_KEY || 'sbp_ae0691ae744bc107ac486303a9bf8d4c4ac77e43'
)

// Database Types for PatternSight Cloud Platform
export interface User {
  id: string
  email: string
  full_name: string
  created_at: string
  updated_at: string
  subscription_tier?: 'starter' | 'pro' | 'elite'
  subscription_status?: 'active' | 'inactive' | 'cancelled' | 'trial'
  subscription_end_date?: string
  daily_analysis_limit: number
  daily_analysis_used: number
  total_analyses: number
  avatar_url?: string
  company?: string
  role?: string
}

export interface Analysis {
  id: string
  user_id: string
  name: string
  type: 'time_series' | 'classification' | 'clustering' | 'anomaly_detection'
  status: 'queued' | 'processing' | 'completed' | 'failed'
  data_source: string
  data_points: number
  accuracy?: number
  confidence: number
  insights: string[]
  results: Record<string, any>
  created_at: string
  completed_at?: string
  processing_time?: number
  model_version: string
}

export interface Dataset {
  id: string
  user_id: string
  name: string
  description?: string
  file_path: string
  file_size: number
  file_type: string
  columns: string[]
  row_count: number
  created_at: string
  updated_at: string
  is_public: boolean
  tags: string[]
}

export interface PatternInsight {
  id: string
  analysis_id: string
  user_id: string
  title: string
  description: string
  type: 'trend' | 'anomaly' | 'correlation' | 'prediction' | 'classification'
  confidence: number
  impact: 'high' | 'medium' | 'low'
  domain: string
  metadata: Record<string, any>
  created_at: string
  is_featured: boolean
}

export interface UsageLog {
  id: string
  user_id: string
  action: 'analysis_created' | 'analysis_completed' | 'dataset_uploaded' | 'insight_generated'
  resource_id: string
  resource_type: 'analysis' | 'dataset' | 'insight'
  metadata: Record<string, any>
  created_at: string
}

// Auth helper functions
export const getCurrentUser = async () => {
  const { data: { user } } = await supabase.auth.getUser()
  return user
}

export const signOut = async () => {
  const { error } = await supabase.auth.signOut()
  return { error }
}

export const signInWithEmail = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password
  })
  return { data, error }
}

export const signUpWithEmail = async (email: string, password: string, fullName: string) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: {
        full_name: fullName
      }
    }
  })
  return { data, error }
}

export const getUserProfile = async (userId: string) => {
  const { data, error } = await supabase
    .from('users')
    .select('*')
    .eq('id', userId)
    .single()
  return { data, error }
}

export const getUserAnalyses = async (userId: string, limit = 10) => {
  const { data, error } = await supabase
    .from('analyses')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false })
    .limit(limit)
  return { data, error }
}

