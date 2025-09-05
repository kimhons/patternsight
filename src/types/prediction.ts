/**
 * PatternSight v4.0 - Prediction Type Definitions
 */

export interface PredictionRequest {
  lotteryType: 'powerball' | 'megamillions' | 'cash4life';
  userId?: string;
  addons?: string[];
}

export interface PillarResult {
  name: string;
  score: number;
  confidence: number;
  predictions: number[];
  explanation: string;
}

export interface PredictionResponse {
  id: string;
  numbers: number[];
  powerball?: number;
  megaBall?: number;
  confidence: number;
  pillarScores: Record<string, number>;
  explanation: string;
  timestamp: string;
  lotteryType: string;
  addonsUsed: string[];
  userId?: string;
}

export interface HistoricalDraw {
  date: string;
  numbers: number[];
  powerball?: number;
  megaBall?: number;
  jackpot?: number;
}

export interface PredictionAnalysis {
  accuracy: number;
  hitRate: number;
  averageMatches: number;
  bestPillar: string;
  worstPillar: string;
  recommendations: string[];
}

