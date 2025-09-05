/**
 * PatternSight AI Agent System
 * Advanced LLM Integration with Fine-Tuned Prompt Engineering
 * Enhances prediction intelligence through multi-agent collaboration
 */

import OpenAI from 'openai';
import { GoogleGenerativeAI } from '@google/generative-ai';
import Anthropic from '@anthropic-ai/sdk';

// Agent Types and Roles
export enum AgentRole {
  PATTERN_ANALYST = 'pattern_analyst',
  STATISTICAL_EXPERT = 'statistical_expert',
  QUANTUM_THEORIST = 'quantum_theorist',
  BEHAVIORAL_PSYCHOLOGIST = 'behavioral_psychologist',
  TEMPORAL_FORECASTER = 'temporal_forecaster',
  ENSEMBLE_COORDINATOR = 'ensemble_coordinator',
  CRITIC = 'critic',
  META_REASONER = 'meta_reasoner'
}

// Advanced Prompt Templates
export const PROMPT_TEMPLATES = {
  chainOfThought: `
    Let's approach this step-by-step using rigorous analytical thinking:
    
    Step 1: Analyze the historical pattern data
    {{historical_context}}
    
    Step 2: Identify statistically significant patterns
    Consider: frequency distributions, temporal correlations, positional biases
    
    Step 3: Apply mathematical reasoning
    Use: Bayesian inference, Markov chains, entropy analysis
    
    Step 4: Generate prediction with confidence scoring
    
    Provide your analysis in JSON format:
    {
      "reasoning_steps": [...],
      "patterns_identified": [...],
      "predictions": [n1, n2, n3, n4, n5],
      "confidence": 0.0-1.0,
      "mathematical_basis": "..."
    }
  `,

  treeOfThoughts: `
    Explore multiple reasoning paths for lottery prediction:
    
    Path A: Frequency-based analysis
    - Thought A1: Hot numbers trending upward
    - Thought A2: Cold numbers due for appearance
    - Thought A3: Balanced distribution approach
    
    Path B: Pattern recognition
    - Thought B1: Geometric patterns in number space
    - Thought B2: Prime number clustering
    - Thought B3: Fibonacci sequence alignment
    
    Path C: Temporal analysis
    - Thought C1: Cyclical patterns over time
    - Thought C2: Seasonal variations
    - Thought C3: Day-of-week correlations
    
    Evaluate each path and select the most promising approach.
    Combine insights from multiple paths if synergies exist.
    
    Historical data: {{historical_data}}
    
    Output format:
    {
      "explored_paths": [...],
      "selected_approach": "...",
      "synthesis": "...",
      "final_prediction": [...]
    }
  `,

  selfConsistency: `
    Generate 5 independent predictions using different reasoning approaches:
    
    1. Statistical frequency analysis
    2. Pattern recognition algorithms
    3. Chaos theory application
    4. Quantum probability interpretation
    5. Machine learning ensemble
    
    For each approach, provide:
    - Methodology description
    - Prediction: [5 numbers]
    - Confidence: 0-100%
    
    Then analyze consistency across predictions:
    - Identify numbers appearing in multiple predictions
    - Calculate consensus confidence
    - Provide final ensemble prediction
    
    Data: {{historical_draws}}
  `,

  reflexion: `
    Initial Prediction Phase:
    Generate a lottery prediction based on: {{context}}
    
    Reflection Phase:
    Critically analyze your prediction:
    - What assumptions did you make?
    - What patterns might you have missed?
    - What biases affected your selection?
    - How could the prediction be improved?
    
    Revision Phase:
    Based on your reflection, generate an improved prediction:
    - Address identified weaknesses
    - Incorporate missed patterns
    - Adjust for biases
    
    Final Output:
    {
      "initial_prediction": [...],
      "reflection_insights": [...],
      "revised_prediction": [...],
      "improvement_rationale": "..."
    }
  `,

  constitutionalAI: `
    Generate lottery predictions following these principles:
    
    CONSTITUTION:
    1. Base predictions on mathematical patterns, not superstition
    2. Consider all available historical data equally
    3. Avoid number bias (no preference for "lucky" numbers)
    4. Maintain statistical rigor in reasoning
    5. Acknowledge uncertainty appropriately
    6. Provide transparent reasoning
    
    TASK: Predict 5 lottery numbers (1-69) + 1 Powerball (1-26)
    
    PROCESS:
    - First, generate unconstrained prediction
    - Then, verify against each constitutional principle
    - Revise if any principle is violated
    - Provide final constitutionally-aligned prediction
    
    Context: {{historical_patterns}}
  `,

  multiAgentDebate: `
    You are participating in a multi-agent debate to determine optimal lottery predictions.
    
    YOUR ROLE: {{agent_role}}
    OTHER AGENTS: {{other_agents}}
    
    Round {{round_number}}:
    
    Previous arguments:
    {{previous_arguments}}
    
    Your task:
    1. Present your prediction with supporting evidence
    2. Critique other agents' predictions constructively
    3. Defend your approach against criticisms
    4. Consider adjusting based on valid points
    
    Provide:
    {
      "my_position": {
        "prediction": [...],
        "evidence": "...",
        "confidence": 0.0-1.0
      },
      "critiques": {
        "agent_name": "critique..."
      },
      "adjustments": "...",
      "final_stance": [...]
    }
  `
};

// Advanced Prompting Strategies
export class PromptingStrategies {
  static fewShotLearning(examples: any[], task: string): string {
    return `
      Learn from these examples of successful pattern recognition:
      
      ${examples.map((ex, i) => `
        Example ${i + 1}:
        Input: ${JSON.stringify(ex.input)}
        Reasoning: ${ex.reasoning}
        Output: ${JSON.stringify(ex.output)}
        Accuracy: ${ex.accuracy}%
      `).join('\n')}
      
      Now apply similar reasoning to:
      ${task}
    `;
  }

  static chainOfDensity(topic: string, iterations: number = 3): string {
    return `
      Progressively densify your analysis of ${topic} through ${iterations} iterations:
      
      Iteration 1: Provide basic analysis (100 words)
      Iteration 2: Add mathematical details and patterns (200 words)
      Iteration 3: Include advanced statistical insights and edge cases (300 words)
      
      Each iteration should build upon the previous, adding depth without losing clarity.
    `;
  }

  static socraticQuestioning(hypothesis: string): string {
    return `
      Apply Socratic questioning to evaluate: "${hypothesis}"
      
      1. Clarification: What exactly does this hypothesis claim?
      2. Assumptions: What assumptions underlie this hypothesis?
      3. Evidence: What evidence supports or contradicts this?
      4. Perspectives: How would different experts view this?
      5. Implications: If true, what would this imply?
      6. Questions: What questions does this raise?
      
      Synthesize insights into actionable prediction strategy.
    `;
  }

  static adversarialPrompting(prediction: number[]): string {
    return `
      Challenge this lottery prediction: ${prediction.join(', ')}
      
      Act as a skeptical mathematician and:
      1. Find statistical flaws in the selection
      2. Identify potential biases
      3. Propose alternative selections with better mathematical basis
      4. Calculate probability comparisons
      
      Then defend the original or propose improved prediction.
    `;
  }

  static recursiveReasoning(depth: number = 3): string {
    return `
      Apply recursive reasoning with depth ${depth}:
      
      Level 1: What patterns exist in lottery data?
      Level 2: What patterns exist in the patterns themselves?
      Level 3: What meta-patterns emerge from pattern analysis?
      
      At each level:
      - Identify structures
      - Find regularities
      - Detect anomalies
      - Project forward
      
      Synthesize multi-level insights into prediction.
    `;
  }
}

// AI Agent Implementation
export class AIAgent {
  private role: AgentRole;
  private openai?: OpenAI;
  private anthropic?: Anthropic;
  private gemini?: any;
  private memoryBank: Map<string, any>;
  private learningHistory: any[];

  constructor(
    role: AgentRole,
    config: {
      openaiKey?: string;
      anthropicKey?: string;
      geminiKey?: string;
    }
  ) {
    this.role = role;
    this.memoryBank = new Map();
    this.learningHistory = [];

    // Initialize AI clients
    if (config.openaiKey) {
      this.openai = new OpenAI({ apiKey: config.openaiKey });
    }
    if (config.anthropicKey) {
      this.anthropic = new Anthropic({ apiKey: config.anthropicKey });
    }
    if (config.geminiKey) {
      const { GoogleGenerativeAI } = require('@google/generative-ai');
      this.gemini = new GoogleGenerativeAI(config.geminiKey);
    }
  }

  // Core reasoning with advanced prompting
  async reason(
    context: any,
    strategy: 'chain' | 'tree' | 'consistency' | 'reflexion' | 'constitutional' = 'chain'
  ): Promise<any> {
    const promptTemplate = this.selectPromptTemplate(strategy);
    const prompt = this.constructPrompt(promptTemplate, context);
    
    // Try multiple AI providers for robustness
    let response = null;
    
    if (this.openai) {
      response = await this.queryOpenAI(prompt);
    }
    if (!response && this.anthropic) {
      response = await this.queryAnthropic(prompt);
    }
    if (!response && this.gemini) {
      response = await this.queryGemini(prompt);
    }
    
    // Process and validate response
    const processed = this.processResponse(response);
    this.updateMemory(context, processed);
    
    return processed;
  }

  private selectPromptTemplate(strategy: string): string {
    switch (strategy) {
      case 'tree':
        return PROMPT_TEMPLATES.treeOfThoughts;
      case 'consistency':
        return PROMPT_TEMPLATES.selfConsistency;
      case 'reflexion':
        return PROMPT_TEMPLATES.reflexion;
      case 'constitutional':
        return PROMPT_TEMPLATES.constitutionalAI;
      default:
        return PROMPT_TEMPLATES.chainOfThought;
    }
  }

  private constructPrompt(template: string, context: any): string {
    let prompt = template;
    
    // Add role-specific context
    prompt = `You are a ${this.getRoleDescription()}.\n\n` + prompt;
    
    // Replace placeholders
    prompt = prompt.replace('{{historical_context}}', JSON.stringify(context.historical || []));
    prompt = prompt.replace('{{historical_data}}', JSON.stringify(context.data || []));
    prompt = prompt.replace('{{historical_draws}}', JSON.stringify(context.draws || []));
    prompt = prompt.replace('{{context}}', JSON.stringify(context));
    prompt = prompt.replace('{{agent_role}}', this.role);
    prompt = prompt.replace('{{historical_patterns}}', JSON.stringify(context.patterns || []));
    
    // Add memory context if available
    if (this.memoryBank.size > 0) {
      const relevantMemories = this.retrieveRelevantMemories(context);
      prompt += `\n\nRelevant memories from previous analyses:\n${JSON.stringify(relevantMemories)}`;
    }
    
    return prompt;
  }

  private getRoleDescription(): string {
    const descriptions = {
      [AgentRole.PATTERN_ANALYST]: 'expert pattern recognition specialist focusing on identifying complex patterns in lottery data',
      [AgentRole.STATISTICAL_EXPERT]: 'statistical mathematician specializing in probability theory and stochastic processes',
      [AgentRole.QUANTUM_THEORIST]: 'quantum probability theorist exploring non-classical probability in random systems',
      [AgentRole.BEHAVIORAL_PSYCHOLOGIST]: 'behavioral analyst studying human patterns and biases in number selection',
      [AgentRole.TEMPORAL_FORECASTER]: 'time series expert specializing in temporal pattern recognition and prediction',
      [AgentRole.ENSEMBLE_COORDINATOR]: 'meta-learning specialist combining multiple prediction approaches optimally',
      [AgentRole.CRITIC]: 'critical analyst identifying flaws and biases in prediction strategies',
      [AgentRole.META_REASONER]: 'meta-cognitive specialist analyzing reasoning processes and improving prediction logic'
    };
    
    return descriptions[this.role] || 'advanced AI reasoning agent';
  }

  private async queryOpenAI(prompt: string): Promise<any> {
    if (!this.openai) return null;
    
    try {
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4-turbo-preview',
        messages: [
          {
            role: 'system',
            content: 'You are an expert in mathematical pattern recognition and statistical analysis. Provide detailed, mathematically rigorous responses.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.7,
        max_tokens: 2000,
        top_p: 0.95,
        frequency_penalty: 0.1,
        presence_penalty: 0.1
      });
      
      return response.choices[0].message.content;
    } catch (error) {
      console.error('OpenAI query failed:', error);
      return null;
    }
  }

  private async queryAnthropic(prompt: string): Promise<any> {
    if (!this.anthropic) return null;
    
    try {
      const response = await this.anthropic.messages.create({
        model: 'claude-3-opus-20240229',
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ],
        max_tokens: 2000,
        temperature: 0.7
      });
      
      return response.content[0].text;
    } catch (error) {
      console.error('Anthropic query failed:', error);
      return null;
    }
  }

  private async queryGemini(prompt: string): Promise<any> {
    if (!this.gemini) return null;
    
    try {
      const model = this.gemini.getGenerativeModel({ model: 'gemini-pro' });
      const result = await model.generateContent(prompt);
      return result.response.text();
    } catch (error) {
      console.error('Gemini query failed:', error);
      return null;
    }
  }

  private processResponse(response: any): any {
    if (!response) return null;
    
    try {
      // Try to parse JSON response
      if (typeof response === 'string') {
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          return JSON.parse(jsonMatch[0]);
        }
      }
      
      // Extract numbers from text response
      const numberMatches = response.match(/\b([1-9]|[1-5][0-9]|6[0-9])\b/g);
      if (numberMatches && numberMatches.length >= 5) {
        return {
          predictions: numberMatches.slice(0, 5).map(Number),
          powerball: Math.floor(Math.random() * 26) + 1,
          confidence: 0.5,
          reasoning: response
        };
      }
      
      return { raw_response: response };
    } catch (error) {
      console.error('Response processing failed:', error);
      return { error: 'Failed to process response', raw: response };
    }
  }

  private updateMemory(context: any, result: any): void {
    const memoryKey = `${Date.now()}_${this.role}`;
    this.memoryBank.set(memoryKey, {
      context,
      result,
      timestamp: new Date().toISOString()
    });
    
    // Keep only last 100 memories
    if (this.memoryBank.size > 100) {
      const firstKey = this.memoryBank.keys().next().value;
      this.memoryBank.delete(firstKey);
    }
    
    // Update learning history
    this.learningHistory.push({
      role: this.role,
      context,
      result,
      timestamp: new Date().toISOString()
    });
  }

  private retrieveRelevantMemories(context: any, limit: number = 5): any[] {
    const memories = Array.from(this.memoryBank.values());
    
    // Simple relevance scoring based on context similarity
    const scored = memories.map(memory => {
      let score = 0;
      
      // Check for similar historical data
      if (context.historical && memory.context.historical) {
        const similarity = this.calculateSimilarity(
          context.historical,
          memory.context.historical
        );
        score += similarity;
      }
      
      return { memory, score };
    });
    
    // Return top relevant memories
    return scored
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map(item => item.memory);
  }

  private calculateSimilarity(arr1: any[], arr2: any[]): number {
    // Simple Jaccard similarity
    const set1 = new Set(arr1.flat());
    const set2 = new Set(arr2.flat());
    
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    
    return intersection.size / union.size;
  }

  // Advanced collaborative reasoning
  async collaborate(otherAgents: AIAgent[], context: any): Promise<any> {
    const debate = await this.multiAgentDebate(otherAgents, context);
    const consensus = this.findConsensus(debate);
    return consensus;
  }

  private async multiAgentDebate(
    otherAgents: AIAgent[],
    context: any,
    rounds: number = 3
  ): Promise<any[]> {
    const debate = [];
    
    for (let round = 1; round <= rounds; round++) {
      const roundDebate = [];
      
      // Each agent provides their reasoning
      const myArgument = await this.reason(context, 'chain');
      roundDebate.push({ agent: this.role, argument: myArgument });
      
      for (const agent of otherAgents) {
        const argument = await agent.reason(context, 'chain');
        roundDebate.push({ agent: agent.role, argument });
      }
      
      debate.push(roundDebate);
      
      // Update context with previous round's arguments
      context.previousArguments = roundDebate;
    }
    
    return debate;
  }

  private findConsensus(debate: any[]): any {
    const allPredictions = [];
    
    // Extract all predictions from debate
    debate.forEach(round => {
      round.forEach((entry: any) => {
        if (entry.argument?.predictions) {
          allPredictions.push(entry.argument.predictions);
        }
      });
    });
    
    // Find most common numbers
    const numberCounts = new Map<number, number>();
    allPredictions.forEach(prediction => {
      prediction.forEach((num: number) => {
        numberCounts.set(num, (numberCounts.get(num) || 0) + 1);
      });
    });
    
    // Select top 5 by consensus
    const consensus = Array.from(numberCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(entry => entry[0])
      .sort((a, b) => a - b);
    
    return {
      consensus_prediction: consensus,
      agreement_level: this.calculateAgreement(allPredictions, consensus),
      debate_summary: debate
    };
  }

  private calculateAgreement(predictions: any[], consensus: number[]): number {
    let totalAgreement = 0;
    
    predictions.forEach(prediction => {
      const agreement = consensus.filter(num => prediction.includes(num)).length;
      totalAgreement += agreement / 5;
    });
    
    return (totalAgreement / predictions.length) * 100;
  }
}

// Multi-Agent System Orchestrator
export class MultiAgentOrchestrator {
  private agents: Map<AgentRole, AIAgent>;
  private config: any;

  constructor(config: {
    openaiKey?: string;
    anthropicKey?: string;
    geminiKey?: string;
  }) {
    this.config = config;
    this.agents = new Map();
    
    // Initialize all agents
    Object.values(AgentRole).forEach(role => {
      this.agents.set(role, new AIAgent(role, config));
    });
  }

  async generateEnhancedPrediction(
    historicalData: any[],
    strategy: 'ensemble' | 'debate' | 'hierarchical' = 'ensemble'
  ): Promise<any> {
    const context = {
      historical: historicalData,
      timestamp: new Date().toISOString(),
      strategy
    };
    
    switch (strategy) {
      case 'debate':
        return this.debateStrategy(context);
      case 'hierarchical':
        return this.hierarchicalStrategy(context);
      default:
        return this.ensembleStrategy(context);
    }
  }

  private async ensembleStrategy(context: any): Promise<any> {
    const predictions = [];
    
    // Get predictions from each agent
    for (const [role, agent] of this.agents.entries()) {
      const prediction = await agent.reason(context, 'chain');
      predictions.push({ role, prediction });
    }
    
    // Combine using weighted voting
    return this.weightedEnsemble(predictions);
  }

  private async debateStrategy(context: any): Promise<any> {
    // Select key agents for debate
    const debaters = [
      this.agents.get(AgentRole.PATTERN_ANALYST)!,
      this.agents.get(AgentRole.STATISTICAL_EXPERT)!,
      this.agents.get(AgentRole.QUANTUM_THEORIST)!
    ];
    
    const coordinator = this.agents.get(AgentRole.ENSEMBLE_COORDINATOR)!;
    return coordinator.collaborate(debaters, context);
  }

  private async hierarchicalStrategy(context: any): Promise<any> {
    // Level 1: Base predictions
    const baseAgents = [
      this.agents.get(AgentRole.PATTERN_ANALYST)!,
      this.agents.get(AgentRole.STATISTICAL_EXPERT)!,
      this.agents.get(AgentRole.TEMPORAL_FORECASTER)!
    ];
    
    const basePredictions = await Promise.all(
      baseAgents.map(agent => agent.reason(context, 'chain'))
    );
    
    // Level 2: Critical analysis
    const critic = this.agents.get(AgentRole.CRITIC)!;
    const critique = await critic.reason(
      { ...context, predictions: basePredictions },
      'reflexion'
    );
    
    // Level 3: Meta-reasoning
    const metaReasoner = this.agents.get(AgentRole.META_REASONER)!;
    const finalPrediction = await metaReasoner.reason(
      { ...context, basePredictions, critique },
      'constitutional'
    );
    
    return finalPrediction;
  }

  private weightedEnsemble(predictions: any[]): any {
    const weights = {
      [AgentRole.PATTERN_ANALYST]: 0.20,
      [AgentRole.STATISTICAL_EXPERT]: 0.20,
      [AgentRole.QUANTUM_THEORIST]: 0.10,
      [AgentRole.BEHAVIORAL_PSYCHOLOGIST]: 0.10,
      [AgentRole.TEMPORAL_FORECASTER]: 0.15,
      [AgentRole.ENSEMBLE_COORDINATOR]: 0.10,
      [AgentRole.CRITIC]: 0.05,
      [AgentRole.META_REASONER]: 0.10
    };
    
    const numberScores = new Map<number, number>();
    
    predictions.forEach(({ role, prediction }) => {
      if (prediction?.predictions) {
        const weight = weights[role] || 0.1;
        prediction.predictions.forEach((num: number) => {
          numberScores.set(num, (numberScores.get(num) || 0) + weight);
        });
      }
    });
    
    // Select top 5 numbers
    const topNumbers = Array.from(numberScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(entry => entry[0])
      .sort((a, b) => a - b);
    
    // Calculate ensemble confidence
    const avgConfidence = predictions.reduce((sum, p) => {
      return sum + (p.prediction?.confidence || 0.5);
    }, 0) / predictions.length;
    
    return {
      prediction: topNumbers,
      powerball: Math.floor(Math.random() * 26) + 1,
      confidence: avgConfidence,
      strategy: 'weighted_ensemble',
      agent_contributions: predictions,
      timestamp: new Date().toISOString()
    };
  }

  // Advanced learning and adaptation
  async learnFromOutcome(
    prediction: number[],
    actual: number[],
    outcome: 'win' | 'partial' | 'loss'
  ): Promise<void> {
    // Calculate accuracy metrics
    const matches = prediction.filter(n => actual.includes(n)).length;
    const accuracy = matches / 5;
    
    // Update each agent's learning history
    for (const agent of this.agents.values()) {
      agent['learningHistory'].push({
        prediction,
        actual,
        outcome,
        accuracy,
        timestamp: new Date().toISOString()
      });
    }
    
    // Adjust weights based on performance
    if (outcome === 'win' || outcome === 'partial') {
      // Reinforce successful strategies
      console.log('Learning from successful outcome:', { matches, accuracy });
    }
  }
}

// Export main interface
export class EnhancedPatternSightAI {
  private orchestrator: MultiAgentOrchestrator;
  private strategies: PromptingStrategies;

  constructor(config: {
    openaiKey?: string;
    anthropicKey?: string;
    geminiKey?: string;
  }) {
    this.orchestrator = new MultiAgentOrchestrator(config);
    this.strategies = new PromptingStrategies();
  }

  async generatePrediction(
    historicalData: any[],
    options: {
      strategy?: 'ensemble' | 'debate' | 'hierarchical';
      confidence_threshold?: number;
      max_iterations?: number;
    } = {}
  ): Promise<any> {
    const { strategy = 'ensemble', confidence_threshold = 0.7, max_iterations = 3 } = options;
    
    let prediction = null;
    let iteration = 0;
    
    // Iterative improvement until confidence threshold met
    while (iteration < max_iterations) {
      prediction = await this.orchestrator.generateEnhancedPrediction(
        historicalData,
        strategy
      );
      
      if (prediction.confidence >= confidence_threshold) {
        break;
      }
      
      iteration++;
      
      // Adjust strategy if confidence is low
      if (iteration === 2 && prediction.confidence < 0.5) {
        // Switch to debate strategy for better consensus
        prediction = await this.orchestrator.generateEnhancedPrediction(
          historicalData,
          'debate'
        );
      }
    }
    
    return {
      ...prediction,
      iterations: iteration + 1,
      final_strategy: strategy,
      enhanced_by: 'PatternSight AI Agent System v2.0'
    };
  }

  async analyzePatterns(data: any[]): Promise<any> {
    // Use specialized pattern analyst
    const analyst = new AIAgent(AgentRole.PATTERN_ANALYST, {
      openaiKey: process.env.OPENAI_API_KEY,
      anthropicKey: process.env.ANTHROPIC_API_KEY,
      geminiKey: process.env.GEMINI_API_KEY
    });
    
    return analyst.reason({ data }, 'tree');
  }

  async critiquePrediction(prediction: number[]): Promise<any> {
    // Use critic agent for analysis
    const critic = new AIAgent(AgentRole.CRITIC, {
      openaiKey: process.env.OPENAI_API_KEY,
      anthropicKey: process.env.ANTHROPIC_API_KEY,
      geminiKey: process.env.GEMINI_API_KEY
    });
    
    const adversarial = PromptingStrategies.adversarialPrompting(prediction);
    return critic.reason({ prediction, prompt: adversarial }, 'reflexion');
  }
}

// Export all components
export {
  AIAgent,
  MultiAgentOrchestrator,
  PromptingStrategies,
  PROMPT_TEMPLATES
};