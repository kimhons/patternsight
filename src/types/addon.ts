/**
 * PatternSight v4.0 - Add-on Type Definitions
 */

export interface AddonConfig {
  id: string;
  name: string;
  description: string;
  price: number;
  features: string[];
  enabled: boolean;
}

export interface CosmicIntelligenceConfig extends AddonConfig {
  celestialBodies: string[];
  astrologicalFactors: string[];
  cosmicAlignment: boolean;
}

export interface ClaudeNexusConfig extends AddonConfig {
  aiEngines: string[];
  reasoningDepth: number;
  multiModelAnalysis: boolean;
}

export interface PremiumEnhancementConfig extends AddonConfig {
  aiModels: string[];
  advancedFeatures: string[];
  exclusiveAccess: boolean;
}

export interface AddonResult {
  addonId: string;
  enhancement: number;
  additionalPredictions?: number[];
  insights: string[];
  confidence: number;
}

