"""
    Figlewski

Limits-to-Arbitrage Lab: reenact Figlewski (1989) and compare with a unified agent
using GJR-GARCH, SMC-ABC, OHMC, quadratic loss, and CVaR risk limits.
Uses Prezo for options, Greeks, volatility, and hedging.
"""
module Figlewski

using Prezo
using Random
using Statistics

include("environment.jl")
include("execution.jl")
include("metrics.jl")
include("cvar.jl")
include("baseline_agent.jl")
include("garch_paths.jl")
include("unified_agent.jl")
include("experiments.jl")

export
    # Environment
    EnvState,
    EnvConfig,
    TradingEnv,
    CostModel,
    observe,
    step!,
    reset!,
    generate_paths,
    # Execution
    execute_hedge!,
    round_to_lot,
    compute_cost,
    compute_cost_state_dependent,  # Upgrade H
    # Metrics
    ExperimentResult,
    compare_agents,
    # CVaR
    compute_cvar,
    size_by_cvar,
    # Baseline
    FiglewskiAgent,
    observe_baseline,
    decide_action_baseline,
    # GARCH paths
    asset_paths_col_garch,
    # Unified agent (full 8-step loop)
    UnifiedAgent,
    Observation,
    Feedback,
    AgentState,
    observe_full,
    belief_update!,
    forecast_risk!,
    decide_action,
    score_step!,
    size_by_cvar_forecast,
    execute_trade!,
    update_feedback!,
    append_return!,
    step_unified!,
    # Event-driven rebalancing (Upgrade E)
    should_rebalance,
    update_trade_state!,
    # Gamma correction (Upgrade D)
    apply_gamma_correction,
    # Belief consistency (Upgrade F)
    apply_belief_consistency,
    # Huberized loss (Upgrade I)
    huber_loss,
    # Unified agent (legacy API)
    observe_unified,
    update_belief_unified,
    decide_action_unified,
    assimilate_and_decide,
    # Experiments
    ExperimentConfig,
    run_experiment,
    run_comparison,
    run_experiment_full_loop
end
