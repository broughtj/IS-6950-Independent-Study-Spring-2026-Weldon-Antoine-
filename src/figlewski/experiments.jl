"""
Experiment runner: run_experiment for one agent, run_comparison for baseline vs unified.
"""

struct ExperimentConfig
    n_paths::Int
    T::Float64
    n_steps::Int
    cost_levels::Vector{Symbol}
    rebalance_freqs::Vector{Int}
    vol_regimes::Vector{Symbol}
    S0::Float64
    r::Float64
    true_vol::Float64
end

function ExperimentConfig(;
    n_paths = 1000,
    T = 1.0,
    n_steps = 252,
    cost_levels = [:low, :medium, :high],
    rebalance_freqs = [1, 2, 5],
    vol_regimes = [:constant],
    S0 = 100.0,
    r = 0.05,
    true_vol = 0.2,
)
    ExperimentConfig(
        n_paths, T, n_steps, cost_levels, rebalance_freqs, vol_regimes,
        S0, r, true_vol,
    )
end

function _cost_model(level::Symbol)
    level === :low && return CostModel(0.0005, 0.0)
    level === :medium && return CostModel(0.001, 1.0)
    level === :high && return CostModel(0.002, 2.0)
    CostModel(0.001, 0.0)
end

"""
    run_experiment(agent, env_config::EnvConfig, n_paths::Int;
                  option = nothing, rebalance_freq::Int = 1, use_ohmc = false) -> ExperimentResult

Run n_paths Monte Carlo paths with the given agent and env_config.
agent can be FiglewskiAgent or UnifiedAgent. For FiglewskiAgent, rebalance_freq is used.
If use_ohmc = true and agent is UnifiedAgent with ohmc_config, uses OHMC for decision.
Returns ExperimentResult aggregating PnLs, tracking errors, turnover, etc.
"""
function run_experiment(
    agent,
    env_config::EnvConfig,
    n_paths::Int;
    option = nothing,
    rebalance_freq::Int = 1,
    use_ohmc::Bool = false,
)
    paths = generate_paths(env_config, n_paths)
    env = TradingEnv(env_config; precomputed_paths = paths)
    pnls = Float64[]
    tracking_errors = Float64[]
    total_turnover = 0.0
    total_trades = 0
    constraint_hits = 0
    total_shares_traded = 0.0
    option_value = 0.0  # will be set from C0

    rebal = agent isa FiglewskiAgent ? agent.rebalance_freq : rebalance_freq

    for path_idx in 1:n_paths
        reset!(env, path_idx)
        opt = option !== nothing ? option : (agent isa FiglewskiAgent ? agent.option : agent.option)
        opt === nothing && error("option required")
        data0 = Prezo.MarketData(env_config.S0, env_config.r, env_config.true_vol, env_config.q)
        engine = Prezo.BlackScholes()
        C0 = Prezo.price(opt, engine, data0)
        option_value = C0  # For Figlewski residual vol metric
        cash = C0
        inventory = 0.0
        path_turnover = 0.0
        path_trades = 0
        delta_target = 0.0
        # Initialize agent state for event-driven rebalancing (Upgrade E)
        path_agent_state = agent isa UnifiedAgent ? AgentState(agent.horizon) : nothing

        for t in 0:(env_config.n_steps - 1)
            S_t = paths[t + 1, path_idx]
            env.state = EnvState(S_t, t, inventory, cash, 0.0, path_idx)

            if agent isa FiglewskiAgent
                if t % rebal == 0
                    delta_target = decide_action_baseline(agent, env)[1]
                end
            else
                # Unified agent: use persistent agent_state for EMA smoothing (Upgrade B)
                # Build returns history
                hist = Float64[]
                for k in max(1, t - 50):(t - 1)
                    if k + 1 <= size(paths, 1)
                        r_k = log(paths[k + 1, path_idx] / paths[k, path_idx])
                        push!(hist, r_k)
                    end
                end
                path_agent_state.returns_history = hist
                
                # Belief update and forecast (uses persistent state)
                rng = env_config.rng
                belief_update!(agent, path_agent_state, rng)
                forecast_risk!(agent, path_agent_state)
                
                # Decide action with EMA smoothing (uses persistent state)
                delta_target = decide_action(agent, env, path_agent_state; use_ohmc = use_ohmc)
            end

            # Determine whether to trade
            do_trade = false
            if agent isa FiglewskiAgent
                do_trade = (t % rebal == 0) || (t == 0)
            else
                # Unified agent: event-driven rebalancing (Upgrade E)
                do_trade = should_rebalance(agent, path_agent_state, delta_target, t)
            end

            if do_trade
                target_shares = delta_target
                raw_trade = target_shares - inventory
                lot = env_config.lot_size
                shares_traded = round_to_lot(raw_trade, lot)
                
                # Compute cost: state-dependent for Unified (Upgrade H), fixed for baseline
                cost = if agent isa UnifiedAgent && agent.use_state_costs
                    # Use agent's vol forecast (annualized)
                    vol_ann = sqrt(path_agent_state.vol_forecast[1] * 252)
                    compute_cost_state_dependent(
                        shares_traded, S_t, env_config.cost_model,
                        vol_ann, env_config.true_vol;
                        vol_sensitivity = agent.vol_cost_sensitivity,
                        impact_coef = agent.impact_coef,
                    )
                else
                    compute_cost(shares_traded, S_t, env_config.cost_model)
                end
                
                inventory += shares_traded
                # Buy: pay shares*S + cost; Sell: receive -shares*S - cost
                cash -= shares_traded * S_t + cost
                path_turnover += abs(shares_traded) * S_t
                path_trades += (shares_traded != 0 ? 1 : 0)
                total_shares_traded += abs(shares_traded)
                # Update trade state for event-driven rebalancing
                if !(agent isa FiglewskiAgent)
                    update_trade_state!(path_agent_state, delta_target, t)
                end
            end
        end

        S_T = paths[env_config.n_steps + 1, path_idx]
        payoff_val = Prezo.payoff(opt, S_T)
        pnl = cash + inventory * S_T - payoff_val
        push!(pnls, pnl)
        push!(tracking_errors, abs(pnl))
        total_turnover += path_turnover
        total_trades += path_trades
    end

    ExperimentResult(
        pnls,
        tracking_errors,
        total_turnover / max(1, n_paths),
        total_trades,
        constraint_hits;
        alpha_cvar = 0.95,
        option_value = option_value,
        underlying_value = env_config.S0,
        total_shares_traded = total_shares_traded,
    )
end

"""
    run_comparison(baseline_agent, unified_agent, env_config::EnvConfig, n_paths::Int;
                   rebalance_freq = 1) -> NamedTuple

Run both agents and return comparison (result_baseline, result_unified, comparison).
"""
function run_comparison(
    baseline_agent,
    unified_agent,
    env_config::EnvConfig,
    n_paths::Int;
    rebalance_freq = 1,
)
    r_base = run_experiment(baseline_agent, env_config, n_paths; rebalance_freq)
    r_unif = run_experiment(unified_agent, env_config, n_paths; rebalance_freq)
    comp = compare_agents(r_base, r_unif; label_A = "Figlewski", label_B = "Unified")
    (result_baseline = r_base, result_unified = r_unif, comparison = comp)
end

# =============================================================================
# Full 8-step loop experiment (uses step_unified!)
# =============================================================================

"""
    run_experiment_full_loop(agent::UnifiedAgent, env_config::EnvConfig, n_paths::Int;
                             use_ohmc = false, cvar_n_sim = 1000) -> ExperimentResult

Run n_paths Monte Carlo paths using the full 8-step unified loop:
  Observe → Belief (SMC-ABC) → Forecast (GARCH) → Decide (OHMC) →
  Score (quadratic) → Size (CVaR) → Execute → Feedback

Returns ExperimentResult with P&L, tracking errors, turnover, constraint hits, etc.
"""
function run_experiment_full_loop(
    agent::UnifiedAgent,
    env_config::EnvConfig,
    n_paths::Int;
    use_ohmc = false,
    cvar_n_sim = 1000,
)
    paths = generate_paths(env_config, n_paths)
    env = TradingEnv(env_config; precomputed_paths = paths)
    opt = agent.option
    opt === nothing && error("UnifiedAgent.option required")

    pnls = Float64[]
    tracking_errors = Float64[]
    total_turnover = 0.0
    total_trades = 0
    total_constraint_hits = 0
    total_shares_traded = 0.0
    cumulative_scores = Float64[]
    option_value = 0.0

    for path_idx in 1:n_paths
        reset!(env, path_idx)
        agent_state = AgentState(agent.horizon)

        # Initialize: receive option premium as cash
        data0 = Prezo.MarketData(env_config.S0, env_config.r, env_config.true_vol, env_config.q)
        C0 = Prezo.price(opt, Prezo.BlackScholes(), data0)
        option_value = C0
        env.state = EnvState(env_config.S0, 0, 0.0, C0, 0.0, path_idx)

        path_turnover = 0.0
        path_trades = 0
        path_constraint_hits = 0
        prev_S = env_config.S0

        for t in 0:(env_config.n_steps - 1)
            # Update price from path
            S_t = paths[t + 1, path_idx]
            env.state = EnvState(S_t, t, env.state.inventory, env.state.cash, 0.0, path_idx)

            # Append return to history (for GARCH/SMC-ABC)
            if t > 0
                append_return!(agent_state, prev_S, S_t)
            end

            # --- Run full 8-step loop ---
            obs = step_unified!(agent, env, agent_state; use_ohmc = use_ohmc, cvar_n_sim = cvar_n_sim)

            # Track turnover and trades
            shares_traded = abs(obs.last_fill)
            if shares_traded > 0
                path_turnover += shares_traded * obs.S
                path_trades += 1
                total_shares_traded += shares_traded
            end

            # Track constraint hits
            if agent_state.feedback.constraint_hit
                path_constraint_hits += 1
            end

            prev_S = S_t
        end

        # Terminal P&L
        S_T = paths[env_config.n_steps + 1, path_idx]
        payoff_val = Prezo.payoff(opt, S_T)
        final_value = env.state.cash + env.state.inventory * S_T
        pnl = final_value - payoff_val

        push!(pnls, pnl)
        push!(tracking_errors, abs(pnl))
        push!(cumulative_scores, agent_state.feedback.cumulative_loss)
        total_turnover += path_turnover
        total_trades += path_trades
        total_constraint_hits += path_constraint_hits
    end

    ExperimentResult(
        pnls,
        tracking_errors,
        total_turnover / max(1, n_paths),
        total_trades,
        total_constraint_hits;
        alpha_cvar = agent.cvar_alpha,
        option_value = option_value,
        underlying_value = env_config.S0,
        total_shares_traded = total_shares_traded,
    )
end
