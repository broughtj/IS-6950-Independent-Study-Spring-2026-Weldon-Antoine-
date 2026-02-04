# Figlewski Limits-to-Arbitrage Module

This module implements a simulation lab for comparing delta hedging strategies, based on Figlewski (1989) "Options Arbitrage in Imperfect Markets."

## Overview

**Key Insight**: *Arbitrage is not a theorem — it is a trade.*

The module compares two agents:
1. **Figlewski Baseline**: Fixed volatility, Black-Scholes delta hedge, discrete rebalancing
2. **Unified Agent**: GJR-GARCH + ABC-SMC belief → OHMC decision → CVaR sizing + 6 upgrades

## Figlewski (1989) Benchmark Metrics

| Metric | Paper | Baseline | Unified |
|--------|-------|----------|---------|
| Residual Risk | ~6-7% | 6.9% ✓ | 6.64% ✓ |
| Shares/Share Hedged | ~5 | 13.4 | **6.1** ✓ |
| Break-even Spread | >30% option | 4.8% | **9.6%** |

## Module Structure

```
src/figlewski/
├── Figlewski.jl        # Main module, exports
├── environment.jl      # EnvConfig, TradingEnv, generate_paths
├── execution.jl        # round_to_lot, compute_cost, compute_cost_state_dependent
├── baseline_agent.jl   # FiglewskiAgent, decide_action_baseline
├── unified_agent.jl    # UnifiedAgent, 8-step loop, all upgrades
├── cvar.jl             # compute_cvar, size_by_cvar
├── metrics.jl          # ExperimentResult, compare_agents
├── experiments.jl      # run_experiment, event-driven rebalancing
├── garch_paths.jl      # GARCH-driven price path generation
└── README.md           # This file
```

## Implemented Upgrades

The unified agent includes 6 variance reduction upgrades:

| Upgrade | Parameter | Description |
|---------|-----------|-------------|
| **E** | `rebalance_threshold=0.05` | Event-driven rebalancing (only trade when Δh > 5%) |
| **B** | `delta_ema_decay=0.7` | Delta EMA smoothing (stabilizes OHMC) |
| **D** | `gamma_correction_k=0.5` | Gamma correction (h = Δ + k·Γ·S·σ·√Δt) |
| **I** | `huber_delta=2.0` | Huberized loss (robust to outliers) |
| **H** | `use_state_costs=true` | State-dependent costs (spread widens in high vol) |
| **C** | `cv_base_weight=0.4` | Control variate (blend OHMC with BS delta) |
| **F** | `belief_penalty_lambda=0.0` | Belief consistency (**disabled** - hurts performance) |

## Usage

### Quick Start

```julia
using Praxeology.Figlewski
using Prezo

# Environment
env_config = Figlewski.EnvConfig(
    S0 = 100.0, r = 0.05, T = 1.0, n_steps = 252,
    true_vol = 0.20, cost_model = Figlewski.CostModel(0.001, 0.0),
)

option = Prezo.EuropeanCall(100.0, 1.0)

# Baseline agent
baseline = Figlewski.FiglewskiAgent(0.20, 1, option)

# Unified agent with all upgrades
garch = Prezo.GJRGARCH(1e-6, 0.04, 0.08, 0.85)
unified = Figlewski.UnifiedAgent(
    garch;
    option = option,
    rebalance_threshold = 0.05,
    delta_ema_decay = 0.7,
    gamma_correction_k = 0.5,
    huber_delta = 2.0,
    use_state_costs = true,
    cv_base_weight = 0.4,
)

# Run experiments
result_base = Figlewski.run_experiment(baseline, env_config, 500)
result_unif = Figlewski.run_experiment(unified, env_config, 500; use_ohmc = true)

# Compare
comp = Figlewski.compare_agents(result_base, result_unif)
```

### Run Full Script

```bash
julia --project=. scripts/run_figlewski.jl
```

### Interactive Notebook

Open `notebooks/Figlewski_Limits_to_Arbitrage.ipynb`

## The 8-Step Loop (Unified Agent)

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Observe    →  Market data: S_t, C_t, spreads, inventory     │
│  2. Belief     →  GJR-GARCH proposes, ABC-SMC scores            │
│  3. Forecast   →  Posterior mean h_t for variance               │
│  4. Decide     →  OHMC + Control Variate + Gamma correction     │
│  5. Score      →  Huberized loss, update EMA                    │
│  6. Size       →  CVaR constraint                               │
│  7. Execute    →  State-dependent costs                         │
│  8. Feedback   →  P&L → back to step 1                          │
└─────────────────────────────────────────────────────────────────┘
```

## Key Results

With 500 paths, 252 daily steps, 10 bps spread:

| Metric | Baseline | Unified | Improvement |
|--------|----------|---------|-------------|
| Residual Vol | 6.9% | **6.64%** | -4% |
| Turnover | 715 | **327** | **-54%** |
| CVaR (95%) | -13.85 | **-12.47** | +10% |
| Mean P&L | 2.60 | **2.86** | +10% |

## Dependencies

- **Prezo.jl**: Options pricing, Greeks, GARCH, OHMC, ABC-SMC
- **Random**: RNG for Monte Carlo
- **Statistics**: mean, std, var

## References

- Figlewski, S. (1989). *Options Arbitrage in Imperfect Markets.* Journal of Finance.
- Figlewski, S. (2017). *Derivatives Valuation Based on Arbitrage: The Trade is Crucial.*

## Related Files

- `scripts/run_figlewski.jl` - Main experiment script
- `notebooks/Figlewski_Limits_to_Arbitrage.ipynb` - Interactive analysis
- `notes/figlewski_lab_report.qmd` - Comprehensive report
