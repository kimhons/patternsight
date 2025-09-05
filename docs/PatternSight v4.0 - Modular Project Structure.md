# PatternSight v4.0 - Modular Project Structure

## 📁 Root Directory Structure

```
patternsight-v4/
├── 📁 apps/                          # Application modules
│   ├── 📁 web/                       # Main web application
│   ├── 📁 dashboard/                 # Dashboard application
│   └── 📁 admin/                     # Admin panel (future)
├── 📁 packages/                      # Shared packages
│   ├── 📁 ui/                        # Shared UI components
│   ├── 📁 auth/                      # Authentication module
│   ├── 📁 database/                  # Database schemas & utilities
│   ├── 📁 api/                       # API clients & types
│   └── 📁 utils/                     # Shared utilities
├── 📁 features/                      # Feature modules
│   ├── 📁 prediction-engine/         # Core prediction system
│   ├── 📁 addon-marketplace/         # Add-on system
│   ├── 📁 user-management/           # User profiles & settings
│   ├── 📁 subscription-billing/      # Billing & subscriptions
│   └── 📁 analytics-reporting/       # Analytics & reports
├── 📁 services/                      # External services
│   ├── 📁 supabase/                  # Supabase configuration
│   ├── 📁 vercel/                    # Deployment configuration
│   └── 📁 ai-providers/              # AI service integrations
├── 📁 docs/                          # Documentation
├── 📁 scripts/                       # Build & deployment scripts
├── 📁 tests/                         # Test suites
└── 📄 Configuration files
```

## 🏗️ Detailed Module Structure

### 📱 Apps Module (`/apps/`)

#### Web Application (`/apps/web/`)
```
apps/web/
├── 📁 src/
│   ├── 📁 app/                       # Next.js App Router
│   │   ├── 📁 (marketing)/           # Marketing pages group
│   │   │   ├── 📄 page.tsx           # Homepage
│   │   │   ├── 📁 about/             # About page
│   │   │   ├── 📁 features/          # Features page
│   │   │   ├── 📁 pricing/           # Pricing page
│   │   │   └── 📁 research/          # Research page
│   │   ├── 📁 (legal)/               # Legal pages group
│   │   │   ├── 📁 privacy/           # Privacy policy
│   │   │   ├── 📁 terms/             # Terms of service
│   │   │   └── 📁 cookies/           # Cookie policy
│   │   ├── 📁 (support)/             # Support pages group
│   │   │   ├── 📁 help/              # Help center
│   │   │   ├── 📁 contact/           # Contact page
│   │   │   └── 📁 faq/               # FAQ page
│   │   ├── 📁 auth/                  # Authentication pages
│   │   │   ├── 📁 signin/            # Sign in page
│   │   │   ├── 📁 signup/            # Sign up page
│   │   │   ├── 📁 forgot-password/   # Password reset
│   │   │   └── 📁 verify-email/      # Email verification
│   │   └── 📁 api/                   # API routes
│   │       ├── 📁 auth/              # Auth endpoints
│   │       └── 📁 webhooks/          # Webhook handlers
│   ├── 📁 components/                # Web-specific components
│   │   ├── 📁 layout/                # Layout components
│   │   ├── 📁 marketing/             # Marketing components
│   │   └── 📁 forms/                 # Form components
│   ├── 📁 styles/                    # Styling
│   └── 📁 lib/                       # Web-specific utilities
├── 📄 package.json
├── 📄 next.config.js
└── 📄 tailwind.config.js
```

#### Dashboard Application (`/apps/dashboard/`)
```
apps/dashboard/
├── 📁 src/
│   ├── 📁 app/                       # Dashboard App Router
│   │   ├── 📄 layout.tsx             # Dashboard layout
│   │   ├── 📄 page.tsx               # Dashboard home
│   │   ├── 📁 predictions/           # Prediction management
│   │   │   ├── 📄 page.tsx           # Predictions list
│   │   │   ├── 📁 [id]/              # Individual prediction
│   │   │   └── 📁 new/               # Create prediction
│   │   ├── 📁 addons/                # Add-on marketplace
│   │   │   ├── 📄 page.tsx           # Marketplace home
│   │   │   ├── 📁 cosmic/            # Cosmic Intelligence
│   │   │   ├── 📁 claude/            # Claude Nexus
│   │   │   └── 📁 premium/           # Premium Enhancement
│   │   ├── 📁 analytics/             # Analytics dashboard
│   │   ├── 📁 settings/              # User settings
│   │   │   ├── 📄 page.tsx           # General settings
│   │   │   ├── 📁 profile/           # Profile management
│   │   │   ├── 📁 billing/           # Billing settings
│   │   │   └── 📁 preferences/       # User preferences
│   │   └── 📁 api/                   # Dashboard API routes
│   ├── 📁 components/                # Dashboard components
│   │   ├── 📁 charts/                # Chart components
│   │   ├── 📁 tables/                # Data tables
│   │   ├── 📁 widgets/               # Dashboard widgets
│   │   └── 📁 modals/                # Modal dialogs
│   └── 📁 hooks/                     # Dashboard-specific hooks
├── 📄 package.json
└── 📄 next.config.js
```

### 📦 Packages Module (`/packages/`)

#### UI Package (`/packages/ui/`)
```
packages/ui/
├── 📁 src/
│   ├── 📁 components/                # Shared UI components
│   │   ├── 📁 atoms/                 # Basic components
│   │   │   ├── 📄 Button.tsx
│   │   │   ├── 📄 Input.tsx
│   │   │   ├── 📄 Badge.tsx
│   │   │   └── 📄 Spinner.tsx
│   │   ├── 📁 molecules/             # Composite components
│   │   │   ├── 📄 Card.tsx
│   │   │   ├── 📄 Modal.tsx
│   │   │   ├── 📄 Dropdown.tsx
│   │   │   └── 📄 SearchBox.tsx
│   │   └── 📁 organisms/             # Complex components
│   │       ├── 📄 Header.tsx
│   │       ├── 📄 Footer.tsx
│   │       ├── 📄 Sidebar.tsx
│   │       └── 📄 DataTable.tsx
│   ├── 📁 styles/                    # Shared styles
│   │   ├── 📄 globals.css
│   │   ├── 📄 components.css
│   │   └── 📄 themes.css
│   ├── 📁 icons/                     # Custom icons
│   └── 📁 utils/                     # UI utilities
├── 📄 package.json
├── 📄 tailwind.config.js
└── 📄 tsconfig.json
```

#### Authentication Package (`/packages/auth/`)
```
packages/auth/
├── 📁 src/
│   ├── 📁 components/                # Auth components
│   │   ├── 📄 AuthModal.tsx
│   │   ├── 📄 ComprehensiveDisclaimer.tsx
│   │   ├── 📄 SignInForm.tsx
│   │   ├── 📄 SignUpForm.tsx
│   │   └── 📄 PasswordReset.tsx
│   ├── 📁 hooks/                     # Auth hooks
│   │   ├── 📄 useAuth.ts
│   │   ├── 📄 useSession.ts
│   │   └── 📄 usePermissions.ts
│   ├── 📁 contexts/                  # Auth contexts
│   │   └── 📄 AuthContext.tsx
│   ├── 📁 services/                  # Auth services
│   │   ├── 📄 authService.ts
│   │   ├── 📄 sessionService.ts
│   │   └── 📄 permissionService.ts
│   ├── 📁 types/                     # Auth types
│   │   └── 📄 auth.types.ts
│   └── 📁 utils/                     # Auth utilities
│       ├── 📄 validation.ts
│       └── 📄 encryption.ts
├── 📄 package.json
└── 📄 tsconfig.json
```

### 🎯 Features Module (`/features/`)

#### Prediction Engine (`/features/prediction-engine/`)
```
features/prediction-engine/
├── 📁 src/
│   ├── 📁 components/                # Prediction UI components
│   │   ├── 📄 PredictionForm.tsx
│   │   ├── 📄 ResultsDisplay.tsx
│   │   ├── 📄 HistoryTable.tsx
│   │   └── 📄 AccuracyChart.tsx
│   ├── 📁 services/                  # Prediction services
│   │   ├── 📄 predictionService.ts
│   │   ├── 📄 uppsEngine.ts
│   │   ├── 📄 patternAnalysis.ts
│   │   └── 📄 dataProcessor.ts
│   ├── 📁 algorithms/                # Core algorithms
│   │   ├── 📄 enhancedUpps.ts
│   │   ├── 📄 statisticalAnalysis.ts
│   │   ├── 📄 patternRecognition.ts
│   │   └── 📄 probabilityCalculator.ts
│   ├── 📁 models/                    # Data models
│   │   ├── 📄 Prediction.ts
│   │   ├── 📄 LotteryData.ts
│   │   └── 📄 AnalysisResult.ts
│   ├── 📁 hooks/                     # Prediction hooks
│   │   ├── 📄 usePrediction.ts
│   │   ├── 📄 useHistory.ts
│   │   └── 📄 useAnalytics.ts
│   └── 📁 utils/                     # Prediction utilities
│       ├── 📄 dataValidation.ts
│       ├── 📄 formatters.ts
│       └── 📄 calculations.ts
├── 📄 package.json
└── 📄 README.md
```

#### Add-on Marketplace (`/features/addon-marketplace/`)
```
features/addon-marketplace/
├── 📁 src/
│   ├── 📁 components/                # Marketplace UI
│   │   ├── 📄 MarketplaceGrid.tsx
│   │   ├── 📄 AddonCard.tsx
│   │   ├── 📄 SubscriptionModal.tsx
│   │   └── 📄 AddonDetails.tsx
│   ├── 📁 addons/                    # Individual add-ons
│   │   ├── 📁 cosmic-intelligence/
│   │   │   ├── 📄 CosmicPredictor.tsx
│   │   │   ├── 📄 cosmicService.ts
│   │   │   └── 📄 cosmicTypes.ts
│   │   ├── 📁 claude-nexus/
│   │   │   ├── 📄 ClaudePredictor.tsx
│   │   │   ├── 📄 claudeService.ts
│   │   │   └── 📄 claudeTypes.ts
│   │   └── 📁 premium-enhancement/
│   │       ├── 📄 PremiumPredictor.tsx
│   │       ├── 📄 premiumService.ts
│   │       └── 📄 premiumTypes.ts
│   ├── 📁 services/                  # Marketplace services
│   │   ├── 📄 marketplaceService.ts
│   │   ├── 📄 subscriptionService.ts
│   │   └── 📄 addonManager.ts
│   ├── 📁 hooks/                     # Marketplace hooks
│   │   ├── 📄 useAddons.ts
│   │   ├── 📄 useSubscriptions.ts
│   │   └── 📄 useMarketplace.ts
│   └── 📁 types/                     # Marketplace types
│       ├── 📄 addon.types.ts
│       └── 📄 subscription.types.ts
├── 📄 package.json
└── 📄 README.md
```

### 🔧 Services Module (`/services/`)

#### Supabase Configuration (`/services/supabase/`)
```
services/supabase/
├── 📁 migrations/                    # Database migrations
│   ├── 📄 001_initial_schema.sql
│   ├── 📄 002_addon_system.sql
│   ├── 📄 003_user_subscriptions.sql
│   └── 📄 004_analytics_tables.sql
├── 📁 functions/                     # Edge functions
│   ├── 📁 generate-prediction/
│   │   ├── 📄 index.ts
│   │   └── 📄 deno.json
│   ├── 📁 process-subscription/
│   │   ├── 📄 index.ts
│   │   └── 📄 deno.json
│   └── 📁 addon-integration/
│       ├── 📄 index.ts
│       └── 📄 deno.json
├── 📁 types/                         # Database types
│   └── 📄 database.types.ts
├── 📄 config.ts
└── 📄 client.ts
```

## 🌿 Branch Structure

### Main Branches
- `main` - Production-ready code
- `develop` - Integration branch for features
- `staging` - Pre-production testing

### Feature Branches
- `feature/prediction-engine-v4` - Core prediction system
- `feature/addon-marketplace` - Add-on system implementation
- `feature/user-authentication` - Auth system overhaul
- `feature/subscription-billing` - Billing integration
- `feature/dashboard-redesign` - Dashboard improvements
- `feature/legal-compliance` - Disclaimer and legal pages
- `feature/analytics-reporting` - Analytics implementation

### Release Branches
- `release/v4.0.0` - Version 4.0 release preparation
- `release/v4.1.0` - Future minor releases

### Hotfix Branches
- `hotfix/critical-bug-fix` - Emergency fixes
- `hotfix/security-patch` - Security updates

## 📋 Routing Configuration

### Web Application Routes (`/apps/web/`)
```typescript
// Marketing Routes
/                           → Homepage
/about                      → About page
/features                   → Features overview
/pricing                    → Pricing plans
/research                   → Academic research

// Legal Routes
/privacy                    → Privacy policy
/terms                      → Terms of service
/cookies                    → Cookie policy

// Support Routes
/help                       → Help center
/contact                    → Contact form
/faq                        → Frequently asked questions

// Authentication Routes
/auth/signin                → Sign in page
/auth/signup                → Sign up page
/auth/forgot-password       → Password reset
/auth/verify-email          → Email verification
```

### Dashboard Application Routes (`/apps/dashboard/`)
```typescript
// Dashboard Routes
/dashboard                  → Dashboard home
/dashboard/predictions      → Prediction management
/dashboard/predictions/new  → Create new prediction
/dashboard/predictions/[id] → View specific prediction

// Add-on Routes
/dashboard/addons           → Marketplace home
/dashboard/addons/cosmic    → Cosmic Intelligence
/dashboard/addons/claude    → Claude Nexus Intelligence
/dashboard/addons/premium   → Premium Enhancement

// Analytics Routes
/dashboard/analytics        → Analytics overview
/dashboard/analytics/performance → Performance metrics
/dashboard/analytics/usage → Usage statistics

// Settings Routes
/dashboard/settings         → General settings
/dashboard/settings/profile → Profile management
/dashboard/settings/billing → Billing settings
/dashboard/settings/preferences → User preferences
```

## 🧪 Testing Structure

```
tests/
├── 📁 unit/                          # Unit tests
│   ├── 📁 components/                # Component tests
│   ├── 📁 services/                  # Service tests
│   ├── 📁 utils/                     # Utility tests
│   └── 📁 hooks/                     # Hook tests
├── 📁 integration/                   # Integration tests
│   ├── 📁 api/                       # API tests
│   ├── 📁 auth/                      # Authentication tests
│   └── 📁 features/                  # Feature tests
├── 📁 e2e/                           # End-to-end tests
│   ├── 📁 user-flows/                # User journey tests
│   ├── 📁 critical-paths/            # Critical functionality
│   └── 📁 regression/                # Regression tests
└── 📁 performance/                   # Performance tests
    ├── 📁 load/                      # Load testing
    └── 📁 stress/                    # Stress testing
```

## 🚀 Deployment Configuration

### Vercel Configuration (`/services/vercel/`)
```
services/vercel/
├── 📄 vercel.json                    # Main Vercel config
├── 📄 web.vercel.json                # Web app config
├── 📄 dashboard.vercel.json          # Dashboard config
└── 📁 environments/                  # Environment configs
    ├── 📄 production.json
    ├── 📄 staging.json
    └── 📄 development.json
```

## 📚 Documentation Structure

```
docs/
├── 📁 api/                           # API documentation
├── 📁 components/                    # Component documentation
├── 📁 features/                      # Feature documentation
├── 📁 deployment/                    # Deployment guides
├── 📁 development/                   # Development guides
├── 📁 architecture/                  # Architecture decisions
└── 📄 README.md                      # Main documentation
```

## 🔧 Build & Development Scripts

```
scripts/
├── 📄 build.sh                      # Build all applications
├── 📄 dev.sh                        # Start development servers
├── 📄 test.sh                       # Run all tests
├── 📄 lint.sh                       # Lint all code
├── 📄 deploy.sh                     # Deploy to production
├── 📄 migrate.sh                    # Run database migrations
└── 📄 setup.sh                      # Initial project setup
```

This modular structure provides:

✅ **Clear Separation of Concerns** - Each feature is isolated
✅ **Reusable Components** - Shared packages for common functionality
✅ **Scalable Architecture** - Easy to add new features and apps
✅ **Proper Routing** - Organized and logical URL structure
✅ **Comprehensive Testing** - Full test coverage strategy
✅ **Deployment Ready** - Configured for multiple environments
✅ **Documentation** - Well-documented codebase
✅ **Version Control** - Proper branching strategy

