import { createClient } from '@supabase/supabase-js'
import { createBrowserClient } from '@supabase/ssr'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

// Client-side Supabase client
export const supabase = createBrowserClient(supabaseUrl, supabaseAnonKey)

// Server-side Supabase client (for API routes)
export const supabaseAdmin = createClient(
  supabaseUrl,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Database Types
export interface DatabaseUser {
  id: string
  email: string
  full_name: string | null
  display_name: string | null
  user_initials: string | null
  created_at: string
  updated_at: string
  last_login: string
  
  // Disclaimer and Legal
  disclaimer_accepted: boolean
  disclaimer_accepted_at: string | null
  disclaimer_ip_address: string | null
  terms_version: string
  
  // Subscription Information
  subscription_tier: 'starter' | 'pro' | 'elite'
  subscription_status: 'trial' | 'active' | 'inactive' | 'cancelled' | 'past_due'
  subscription_start_date: string | null
  subscription_end_date: string | null
  stripe_customer_id: string | null
  stripe_subscription_id: string | null
  
  // Usage Limits and Tracking
  predictions_used_today: number
  predictions_limit_daily: number
  total_predictions_generated: number
  
  // Profile and Preferences
  birth_date: string | null
  timezone: string
  notification_preferences: Record<string, any>
  
  // Metadata
  metadata: Record<string, any>
  is_active: boolean
}

export interface DatabasePrediction {
  id: string
  user_id: string
  
  // Prediction Data
  numbers: number[]
  powerball: number | null
  mega_ball: number | null
  game_type: 'powerball' | 'mega_millions' | 'lotto'
  state: string
  
  // Scoring System
  statistical_score: number
  llm_score: number
  astronomical_score: number
  numerological_score: number
  sacred_geometry_score: number
  unified_score: number
  
  // AI Insights
  llm_insights: string | null
  confidence_tier: 'low' | 'medium' | 'high' | 'cosmic'
  prediction_method: string
  
  // Targeting and Results
  target_draw_date: string | null
  actual_numbers: number[] | null
  actual_powerball: number | null
  actual_mega_ball: number | null
  matches: number
  prize_amount: number
  is_winner: boolean
  
  // Metadata
  created_at: string
  updated_at: string
  ip_address: string | null
  user_agent: string | null
}

export interface DatabaseSubscription {
  id: string
  user_id: string
  
  // Stripe Integration
  stripe_subscription_id: string | null
  stripe_customer_id: string | null
  stripe_price_id: string | null
  stripe_product_id: string | null
  
  // Subscription Details
  plan_name: string
  plan_type: 'subscription' | 'bundle'
  tier: 'starter' | 'pro' | 'elite'
  billing_cycle: 'monthly' | 'yearly' | 'one_time' | null
  
  // Pricing
  amount: number
  currency: string
  discount_percent: number
  
  // Status and Dates
  status: 'active' | 'inactive' | 'cancelled' | 'past_due' | 'trialing'
  current_period_start: string | null
  current_period_end: string | null
  trial_start: string | null
  trial_end: string | null
  cancelled_at: string | null
  
  // Metadata
  created_at: string
  updated_at: string
}

export interface DatabasePredictionBundle {
  id: string
  user_id: string
  
  // Bundle Details
  bundle_name: string
  predictions_count: number
  predictions_used: number
  
  // Purchase Information
  stripe_payment_intent_id: string | null
  amount_paid: number
  currency: string
  
  // Status and Expiry
  status: 'active' | 'expired' | 'refunded'
  expires_at: string | null
  never_expires: boolean
  
  // Metadata
  created_at: string
  updated_at: string
}

export interface DatabaseUsageLog {
  id: string
  user_id: string | null
  
  // Action Details
  action: string
  resource_type: string | null
  resource_id: string | null
  
  // Request Information
  ip_address: string | null
  user_agent: string | null
  referer: string | null
  
  // Additional Data
  details: Record<string, any>
  success: boolean
  error_message: string | null
  
  // Timing
  created_at: string
  processing_time_ms: number | null
}

export interface DatabaseLotteryResult {
  id: string
  
  // Game Information
  game_type: 'powerball' | 'mega_millions' | 'lotto'
  state: string
  draw_date: string
  
  // Winning Numbers
  numbers: number[]
  powerball: number | null
  mega_ball: number | null
  multiplier: number | null
  
  // Prize Information
  jackpot_amount: number | null
  next_jackpot_amount: number | null
  total_winners: number
  
  // Metadata
  created_at: string
  updated_at: string
}

// Supabase Database Service Class
export class SupabaseService {
  private client = supabase
  private adminClient = supabaseAdmin

  // User Management
  async getUser(userId: string): Promise<DatabaseUser | null> {
    const { data, error } = await this.client
      .from('users')
      .select('*')
      .eq('id', userId)
      .single()
    
    if (error) {
      console.error('Error fetching user:', error)
      return null
    }
    
    return data
  }

  async updateUser(userId: string, updates: Partial<DatabaseUser>): Promise<boolean> {
    const { error } = await this.client
      .from('users')
      .update({ ...updates, updated_at: new Date().toISOString() })
      .eq('id', userId)
    
    if (error) {
      console.error('Error updating user:', error)
      return false
    }
    
    return true
  }

  async acceptDisclaimer(userId: string, userInitials: string, ipAddress?: string): Promise<boolean> {
    const { error } = await this.client
      .from('users')
      .update({
        disclaimer_accepted: true,
        disclaimer_accepted_at: new Date().toISOString(),
        disclaimer_ip_address: ipAddress,
        user_initials: userInitials,
        updated_at: new Date().toISOString()
      })
      .eq('id', userId)
    
    if (error) {
      console.error('Error accepting disclaimer:', error)
      return false
    }
    
    return true
  }

  // Prediction Management
  async canGeneratePrediction(userId: string): Promise<boolean> {
    const { data, error } = await this.adminClient
      .rpc('can_generate_prediction', { user_uuid: userId })
    
    if (error) {
      console.error('Error checking prediction limit:', error)
      return false
    }
    
    return data
  }

  async consumePrediction(userId: string): Promise<boolean> {
    const { data, error } = await this.adminClient
      .rpc('consume_prediction', { user_uuid: userId })
    
    if (error) {
      console.error('Error consuming prediction:', error)
      return false
    }
    
    return data
  }

  async createPrediction(prediction: Omit<DatabasePrediction, 'id' | 'created_at' | 'updated_at'>): Promise<string | null> {
    const { data, error } = await this.client
      .from('predictions')
      .insert(prediction)
      .select('id')
      .single()
    
    if (error) {
      console.error('Error creating prediction:', error)
      return null
    }
    
    return data.id
  }

  async getUserPredictions(userId: string, limit: number = 10): Promise<DatabasePrediction[]> {
    const { data, error } = await this.client
      .from('predictions')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(limit)
    
    if (error) {
      console.error('Error fetching predictions:', error)
      return []
    }
    
    return data || []
  }

  // Subscription Management
  async getUserSubscription(userId: string): Promise<DatabaseSubscription | null> {
    const { data, error } = await this.client
      .from('subscriptions')
      .select('*')
      .eq('user_id', userId)
      .eq('status', 'active')
      .single()
    
    if (error && error.code !== 'PGRST116') { // PGRST116 is "no rows returned"
      console.error('Error fetching subscription:', error)
    }
    
    return data || null
  }

  async createSubscription(subscription: Omit<DatabaseSubscription, 'id' | 'created_at' | 'updated_at'>): Promise<string | null> {
    const { data, error } = await this.client
      .from('subscriptions')
      .insert(subscription)
      .select('id')
      .single()
    
    if (error) {
      console.error('Error creating subscription:', error)
      return null
    }
    
    return data.id
  }

  // Bundle Management
  async getUserBundles(userId: string): Promise<DatabasePredictionBundle[]> {
    const { data, error } = await this.client
      .from('prediction_bundles')
      .select('*')
      .eq('user_id', userId)
      .eq('status', 'active')
      .order('created_at', { ascending: false })
    
    if (error) {
      console.error('Error fetching bundles:', error)
      return []
    }
    
    return data || []
  }

  async createBundle(bundle: Omit<DatabasePredictionBundle, 'id' | 'created_at' | 'updated_at'>): Promise<string | null> {
    const { data, error } = await this.client
      .from('prediction_bundles')
      .insert(bundle)
      .select('id')
      .single()
    
    if (error) {
      console.error('Error creating bundle:', error)
      return null
    }
    
    return data.id
  }

  // Usage Logging
  async logUsage(log: Omit<DatabaseUsageLog, 'id' | 'created_at'>): Promise<void> {
    const { error } = await this.client
      .from('usage_logs')
      .insert(log)
    
    if (error) {
      console.error('Error logging usage:', error)
    }
  }

  // System Settings
  async getSystemSetting(key: string): Promise<any> {
    const { data, error } = await this.client
      .from('system_settings')
      .select('value')
      .eq('key', key)
      .single()
    
    if (error) {
      console.error('Error fetching system setting:', error)
      return null
    }
    
    return data?.value
  }

  // Lottery Results
  async getLatestResults(gameType: string, limit: number = 5): Promise<DatabaseLotteryResult[]> {
    const { data, error } = await this.client
      .from('lottery_results')
      .select('*')
      .eq('game_type', gameType)
      .order('draw_date', { ascending: false })
      .limit(limit)
    
    if (error) {
      console.error('Error fetching lottery results:', error)
      return []
    }
    
    return data || []
  }

  // Admin Functions
  async resetDailyUsage(): Promise<boolean> {
    const { error } = await this.adminClient
      .rpc('reset_daily_usage')
    
    if (error) {
      console.error('Error resetting daily usage:', error)
      return false
    }
    
    return true
  }

  // Analytics and Reporting
  async getUserStats(userId: string): Promise<any> {
    const { data, error } = await this.client
      .from('predictions')
      .select('id, is_winner, prize_amount, created_at')
      .eq('user_id', userId)
    
    if (error) {
      console.error('Error fetching user stats:', error)
      return null
    }
    
    const totalPredictions = data.length
    const totalWins = data.filter(p => p.is_winner).length
    const totalPrizes = data.reduce((sum, p) => sum + (p.prize_amount || 0), 0)
    const winRate = totalPredictions > 0 ? (totalWins / totalPredictions) * 100 : 0
    
    return {
      totalPredictions,
      totalWins,
      totalPrizes,
      winRate: Math.round(winRate * 100) / 100
    }
  }
}

// Export singleton instance
export const supabaseService = new SupabaseService()

