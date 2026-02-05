"""
Unified agent: one agent, one system — a closed-loop controller with Bayesian
state estimation and coherent risk constraints.

**Architecture: GJR-GARCH + ABC-SMC Hybrid**

The key insight is that GJR-GARCH and ABC-SMC work together, not sequentially:
- **GJR-GARCH = state transition / proposal model** for latent volatility
- **ABC-SMC = sequential particle machinery** that scores particles by how well
  simulated returns match observed features, without trusting the GJR likelihood

Particles carry both θ = (ω, α, γ, β) AND h_t (volatility state).
ABC scores using leverage-aware summary statistics:
- s₁: realized variance (Σr²)
- s₂: leverage correlation corr(r_{t-1}, r_t²) — should be negative if leverage exists
- s₃: downside variance share
- s₄, s₅: tail quantiles
- s₆: kurtosis

**The unified loop (end-to-end):**

1. **Observe** market data: S_t, option prices C_t, spreads, fills, inventory.
2. **Belief update (GJR-GARCH + ABC-SMC hybrid):**
   - GJR-GARCH proposes/transitions volatility state h_t
   - ABC-SMC scores particles by summary statistic match (not likelihood)
   - Output: posterior distribution over θ and h_t
3. **Forecast risk:** use posterior mean h_t for conditional variance.
4. **Decide action (OHMC or BS delta):** choose hedge ratios.
5. **Score (quadratic loss):** compute tracking loss, update EMA for policy tuning.
6. **Size position (CVaR):** scale by tail risk + capital constraints.
7. **Execute** under transaction costs, discreteness, slippage.
8. **Feedback:** realized P&L, inventory drift, constraint hits → back to step 1.

This is a **closed-loop controller with Bayesian state estimation**.
"""

using Statistics: mean, std

# =============================================================================
# Step 1: Observation — richer market snapshot
# =============================================================================

"""
    Observation

Full market observation at time t:
- `S`: spot price
- `C`: option mid price (Black-Scholes or market)
- `bid`, `ask`: option bid/ask (derived from spread)
- `last_fill`: shares traded in last execution
- `fill_price`: average fill price
- `inventory`: current share inventory
- `cash`: cash balance
- `t`: step index
- `tau`: time to expiry
"""
struct Observation
    S::Float64
    C::Float64
    bid::Float64
    ask::Float64
    last_fill::Float64
    fill_price::Float64
    inventory::Float64
    cash::Float64
    t::Int
    tau::Float64
end

function observe_full(env::TradingEnv, option, last_fill::Float64, fill_price::Float64;
                      vol_est::Float64 = NaN)
    cfg = env.config
    s = env.state
    tau = max(0.0, cfg.T * (1 - s.t / cfg.n_steps))
    # Use provided vol estimate (from agent's belief), or fall back to true_vol
    σ = isnan(vol_est) ? cfg.true_vol : vol_est
    data = Prezo.MarketData(s.S, cfg.r, σ, cfg.q)

    C_mid = if option isa Prezo.EuropeanCall
        opt_tau = Prezo.EuropeanCall(option.strike, tau)
        Prezo.price(opt_tau, Prezo.BlackScholes(), data)
    elseif option isa Prezo.EuropeanPut
        opt_tau = Prezo.EuropeanPut(option.strike, tau)
        Prezo.price(opt_tau, Prezo.BlackScholes(), data)
    else
        0.0
    end

    half_spread = C_mid * cfg.cost_model.spread_bps
    Observation(
        s.S, C_mid, C_mid - half_spread, C_mid + half_spread,
        last_fill, fill_price, s.inventory, s.cash, s.t, tau,
    )
end

# =============================================================================
# Step 8: Feedback — realized metrics for next iteration
# =============================================================================

"""
    Feedback

Realized outcomes fed back to the loop:
- `realized_pnl`: step P&L (mark-to-market change + execution cost)
- `inventory_drift`: change in inventory from target
- `constraint_hit`: true if CVaR or position limit was binding
- `tracking_loss`: squared replication error this step
- `cumulative_loss`: running sum of tracking losses
"""
mutable struct Feedback
    realized_pnl::Float64
    inventory_drift::Float64
    constraint_hit::Bool
    tracking_loss::Float64
    cumulative_loss::Float64
end

Feedback() = Feedback(0.0, 0.0, false, 0.0, 0.0)

# =============================================================================
# Agent state (mutable, carries history and feedback)
# =============================================================================

"""
    AgentState

Mutable state carried across steps:
- `returns_history`: rolling window of log returns for GARCH/SMC-ABC
- `vol_forecast`: latest variance forecast (vector, horizon steps)
- `feedback`: Feedback struct
- `prev_option_value`: for tracking error calculation
- `score_ema`: exponential moving average of quadratic loss (for adaptive tuning)
- `last_trade_step`: step index of last trade (for event-driven rebalancing)
- `last_target_delta`: target delta at last trade
- `delta_ema`: EMA of OHMC delta for smoothing (Upgrade B variance reduction)
- `posterior_vol_std`: std of posterior vol forecast (Upgrade F belief consistency)
"""
mutable struct AgentState
    returns_history::Vector{Float64}
    vol_forecast::Vector{Float64}
    feedback::Feedback
    prev_option_value::Float64
    score_ema::Float64  # EMA of quadratic loss for policy adaptation
    last_trade_step::Int  # For event-driven rebalancing
    last_target_delta::Float64
    delta_ema::Float64  # EMA of OHMC delta for variance reduction
    posterior_vol_std::Float64  # Posterior uncertainty (Upgrade F)
end

AgentState(horizon::Int) = AgentState(Float64[], fill(0.04, horizon), Feedback(), 0.0, 0.0, -1, 0.0, NaN, 0.0)

# =============================================================================
# Unified Agent struct
# =============================================================================

struct UnifiedAgent{O,G}
    garch_model::G
    ohmc_config::Union{Prezo.OHMCConfig, Nothing}
    cvar_ohmc_config::Union{Any, Nothing}  # Prezo.CVaROHMCConfig when set (Upgrade A)
    abc_method::Union{Prezo.ABCSMC, Nothing}
    cvar_alpha::Float64
    L_max::Float64
    option::O
    horizon::Int
    score_ema_decay::Float64  # decay for score EMA (e.g., 0.95)
    # Event-driven rebalancing (Upgrade E)
    rebalance_threshold::Float64  # Only trade when |Δh| > τ
    max_days_no_trade::Int        # Force trade after this many days
    # Delta smoothing (Upgrade B variance reduction)
    delta_ema_decay::Float64      # EMA decay for smoothing OHMC delta (e.g., 0.7)
    # Gamma correction (Upgrade D)
    gamma_correction_k::Float64   # k in h = Δ + k·Γ·S·σ·√Δt
    # Belief consistency (Upgrade F)
    belief_penalty_lambda::Float64  # Scale delta down when posterior is uncertain
    # Huberized loss (Upgrade I)
    huber_delta::Float64            # Threshold for Huber loss (0 = pure quadratic)
    # State-dependent costs (Upgrade H)
    use_state_costs::Bool           # Enable vol/size-dependent costs
    vol_cost_sensitivity::Float64   # How much spread widens with vol
    impact_coef::Float64            # Market impact coefficient
    # Control variate (Upgrade C)
    use_control_variate::Bool       # Blend OHMC with BS delta
    cv_base_weight::Float64         # Base weight on BS delta (0 = pure OHMC, 1 = pure BS)
    # Kelly for scale (Upgrade A: Option B — OHMC+CVaR decides, Kelly sizes)
    use_kelly_sizing::Bool          # Scale position by fractional Kelly
    kelly_fraction::Float64         # e.g. 0.5 for half-Kelly
end

function UnifiedAgent(
    garch_model;
    ohmc_config = nothing,
    cvar_ohmc_config = nothing,  # Prezo.CVaROHMCConfig for Upgrade A (CVaR inside OHMC)
    abc_method = nothing,
    cvar_alpha = 0.95,
    L_max = Inf,
    option = nothing,
    horizon = 5,
    score_ema_decay = 0.95,
    # Event-driven rebalancing (Upgrade E)
    rebalance_threshold = 0.02,  # Only trade when |Δh| > 2%
    max_days_no_trade = 5,       # Force trade after 5 days
    # Delta smoothing (Upgrade B variance reduction)
    delta_ema_decay = 0.7,       # EMA decay for smoothing OHMC delta
    # Gamma correction (Upgrade D)
    gamma_correction_k = 0.5,    # k in h = Δ + k·Γ·S·σ·√Δt (0.5 is typical)
    # Belief consistency (Upgrade F)
    belief_penalty_lambda = 2.0, # Scale delta by 1/(1 + λ·CV²) where CV = std/mean
    # Huberized loss (Upgrade I)
    huber_delta = 5.0,           # Threshold for Huber loss ($5 = typical option spread)
    # State-dependent costs (Upgrade H)
    use_state_costs = true,      # Enable vol/size-dependent costs
    vol_cost_sensitivity = 2.0,  # Spread widens 2x when vol doubles
    impact_coef = 0.0001,        # Market impact coefficient
    # Control variate (Upgrade C)
    use_control_variate = true,  # Blend OHMC with BS delta
    cv_base_weight = 0.3,        # 30% BS delta, 70% OHMC (adjustable)
    # Kelly for scale (Upgrade A Option B)
    use_kelly_sizing = false,    # Set true to scale by fractional Kelly
    kelly_fraction = 0.5,        # Half-Kelly
)
    UnifiedAgent(
        garch_model, ohmc_config, cvar_ohmc_config, abc_method, cvar_alpha, L_max,
        option, horizon, score_ema_decay,
        rebalance_threshold, max_days_no_trade,
        delta_ema_decay, gamma_correction_k,
        belief_penalty_lambda, huber_delta,
        use_state_costs, vol_cost_sensitivity, impact_coef,
        use_control_variate, cv_base_weight,
        use_kelly_sizing, kelly_fraction
    )
end

# =============================================================================
# Step 2: Belief update — GJR-GARCH + ABC-SMC hybrid
# =============================================================================
#
# Architecture (from user specification):
#   • GJR-GARCH = state transition / proposal model for latent volatility
#   • ABC-SMC = sequential particle machinery that scores particles by
#               how well simulated returns match observed features
#
# Particles carry both θ (ω, α, γ, β) AND h_t (volatility state).
# ABC scores using leverage-aware summary statistics:
#   s₁ = Σr² (realized variance proxy)
#   s₂ = corr(r_{t-1}, r_t²) (leverage correlation, should be negative)
#   s₃ = downside variance share
#   s₄, s₅ = quantiles (tail behavior)
#   s₆ = kurtosis
# =============================================================================

function _garch_unconditional_variance(m)
    if m isa Prezo.GJRGARCH
        denom = 1 - m.α - m.γ / 2 - m.β
        return denom > 0 ? m.ω / denom : m.ω / 0.01
    elseif m isa Prezo.GARCH
        denom = 1 - m.α - m.β
        return denom > 0 ? m.ω / denom : m.ω / 0.01
    else
        return m.ω / max(1 - m.β, 0.01)
    end
end

"""
    gjr_garch_step(r_prev, h_prev, ω, α, γ, β) -> h_t

Single GJR-GARCH(1,1) variance update:
    h_t = ω + (α + γ·1{r<0})·r²_{t-1} + β·h_{t-1}
"""
function gjr_garch_step(r_prev::Float64, h_prev::Float64, ω, α, γ, β)
    leverage = r_prev < 0 ? γ : 0.0
    max(ω + (α + leverage) * r_prev^2 + β * h_prev, 1e-10)
end

"""
    simulate_gjr_window(params, h_start, L, rng) -> Vector{Float64}

Simulate L returns from GJR-GARCH with given parameters.
params = [ω, α, γ, β] (4 elements)
"""
function simulate_gjr_window(params::Vector{Float64}, h_start::Float64, L::Int, rng)
    ω, α, γ, β = params[1], params[2], params[3], params[4]
    returns = Vector{Float64}(undef, L)
    h = h_start
    for t in 1:L
        ε = randn(rng)
        r = sqrt(h) * ε
        returns[t] = r
        h = gjr_garch_step(r, h, ω, α, γ, β)
    end
    returns
end

"""
    leverage_summary_stats(r) -> Vector{Float64}

Compute leverage-aware summary statistics for ABC scoring:
- s₁: realized variance (Σr²)
- s₂: leverage correlation corr(r_{t-1}, r_t²) 
- s₃: downside variance share
- s₄: left tail (1st percentile)
- s₅: right tail (99th percentile)
- s₆: kurtosis
"""
function leverage_summary_stats(r::Vector{Float64})
    n = length(r)
    n < 5 && return [var(r), 0.0, 0.5, minimum(r), maximum(r), 3.0]
    
    # s₁: realized variance
    s1 = sum(r .^ 2)
    
    # s₂: leverage correlation - corr(r_{t-1}, r_t²)
    r_lag = r[1:end-1]
    r_sq = r[2:end] .^ 2
    s2 = length(r_lag) > 2 ? cor(r_lag, r_sq) : 0.0
    isnan(s2) && (s2 = 0.0)
    
    # s₃: downside variance share
    down_mask = r .< 0
    s3 = sum(down_mask) > 0 ? sum(r[down_mask] .^ 2) / max(s1, 1e-10) : 0.5
    
    # s₄, s₅: tail quantiles
    sorted_r = sort(r)
    q_lo = max(1, round(Int, 0.01 * n))
    q_hi = min(n, round(Int, 0.99 * n))
    s4 = sorted_r[q_lo]
    s5 = sorted_r[q_hi]
    
    # s₆: kurtosis
    μ = mean(r)
    σ = std(r)
    s6 = σ > 1e-10 ? mean(((r .- μ) ./ σ) .^ 4) : 3.0
    
    [s1, s2, s3, s4, s5, s6]
end

"""
    enforce_garch_constraints!(params; delta=0.02)

Enforce GJR-GARCH stationarity: α + γ/2 + β < 1 - δ
Modifies params in place.
"""
function enforce_garch_constraints!(params::AbstractVector{Float64}; delta=0.02)
    # Ensure positivity
    params[1] = max(params[1], 1e-8)  # ω > 0
    params[2] = max(params[2], 0.0)   # α ≥ 0
    params[3] = max(params[3], 0.0)   # γ ≥ 0
    params[4] = max(params[4], 0.0)   # β ≥ 0
    
    # Stationarity: α + γ/2 + β < 1 - δ
    c = params[2] + params[3] / 2 + params[4]
    target = 1.0 - delta
    if c >= target
        scale = (target - 0.01) / c
        params[2] *= scale
        params[3] *= scale
        params[4] *= scale
    end
    nothing
end

"""
    belief_update!(agent::UnifiedAgent, agent_state::AgentState, rng)

Hybrid GJR-GARCH + ABC-SMC belief update.

Architecture:
- GJR-GARCH = proposal/transition model for volatility
- ABC-SMC = scoring machinery using leverage-aware summary statistics

Particles carry θ = (ω, α, γ, β) and h_t. ABC scores by comparing
simulated vs observed summary statistics without trusting the likelihood.

Updates agent_state.vol_forecast with the posterior mean variance.
"""
function belief_update!(agent::UnifiedAgent, agent_state::AgentState, rng)
    n_ret = length(agent_state.returns_history)
    # Need enough data for meaningful leverage statistics
    n_ret < 30 && return nothing
    agent.abc_method === nothing && return nothing
    
    # --- Settings ---
    n_particles = 50
    window_len = min(n_ret, 50)  # Rolling window for summaries
    ε_tol = 0.15  # ABC tolerance (adaptive would be better)
    
    # Observed summary statistics
    obs_window = agent_state.returns_history[max(1, n_ret - window_len + 1):n_ret]
    s_obs = leverage_summary_stats(obs_window)
    
    # --- Initialize particles: θ = (ω, α, γ, β), h ---
    # Use agent's GARCH model as prior center
    m = agent.garch_model
    ω0 = m isa Prezo.GJRGARCH ? m.ω : (m isa Prezo.GARCH ? m.ω : 1e-5)
    α0 = m isa Prezo.GJRGARCH ? m.α : (m isa Prezo.GARCH ? m.α : 0.05)
    γ0 = m isa Prezo.GJRGARCH ? m.γ : 0.1
    β0 = m isa Prezo.GJRGARCH ? m.β : (m isa Prezo.GARCH ? m.β : 0.85)
    
    particles = Matrix{Float64}(undef, n_particles, 4)  # [ω, α, γ, β]
    h_particles = Vector{Float64}(undef, n_particles)
    weights = fill(1.0 / n_particles, n_particles)
    
    for i in 1:n_particles
        # Jitter around prior center
        particles[i, 1] = ω0 * exp(0.3 * randn(rng))  # ω
        particles[i, 2] = α0 * exp(0.3 * randn(rng))  # α
        particles[i, 3] = γ0 * exp(0.3 * randn(rng))  # γ
        particles[i, 4] = β0 * exp(0.3 * randn(rng))  # β
        enforce_garch_constraints!(@view particles[i, :])
        
        # Initialize h from sample variance
        h_particles[i] = var(obs_window) * (0.8 + 0.4 * rand(rng))
    end
    
    # --- ABC-SMC scoring ---
    for i in 1:n_particles
        θ = particles[i, :]
        h_start = h_particles[i]
        
        # Simulate pseudo-window under particle's parameters
        sim_returns = simulate_gjr_window(θ, h_start, window_len, rng)
        s_sim = leverage_summary_stats(sim_returns)
        
        # Compute whitened distance (simplified: use element-wise scaling)
        scale = max.(abs.(s_obs), 0.01)
        dist = sqrt(sum(((s_sim .- s_obs) ./ scale) .^ 2))
        
        # ABC kernel weight (Epanechnikov-like)
        if dist < ε_tol
            weights[i] = (1 - (dist / ε_tol)^2)
        else
            weights[i] = 1e-10
        end
    end
    
    # Normalize weights
    w_sum = sum(weights)
    w_sum > 0 && (weights ./= w_sum)
    
    # --- Compute posterior mean and std of volatility (Upgrade F) ---
    # Compute filtered h for each particle
    h_filtered_all = Vector{Float64}(undef, n_particles)
    for i in 1:n_particles
        θ = particles[i, :]
        h_filt = var(obs_window)  # start
        for t in 2:n_ret
            r_prev = agent_state.returns_history[t-1]
            h_filt = gjr_garch_step(r_prev, h_filt, θ[1], θ[2], θ[3], θ[4])
        end
        h_filtered_all[i] = h_filt
    end
    
    # Posterior mean
    h_mean = sum(weights .* h_filtered_all)
    
    # Posterior std (for Upgrade F belief consistency)
    h_var = sum(weights .* (h_filtered_all .- h_mean).^2)
    h_std = sqrt(max(h_var, 0.0))
    
    # Update agent state
    agent_state.vol_forecast .= clamp(h_mean, 1e-8, 4.0)
    agent_state.posterior_vol_std = h_std
    
    nothing
end

# =============================================================================
# Step 3: Forecast risk (GJR-GARCH)
# =============================================================================

"""
    forecast_risk!(agent::UnifiedAgent, agent_state::AgentState)

GARCH forecast: update vol_forecast from returns_history.
"""
function forecast_risk!(agent::UnifiedAgent, agent_state::AgentState)
    hist = agent_state.returns_history
    if length(hist) >= 1
        agent_state.vol_forecast .= Prezo.forecast(agent.garch_model, hist, agent.horizon)
    else
        agent_state.vol_forecast .= _garch_unconditional_variance(agent.garch_model)
    end
    nothing
end

# =============================================================================
# Step 4: Decide action (OHMC or BS delta)
# =============================================================================

"""
    decide_action(agent::UnifiedAgent, env::TradingEnv, agent_state::AgentState;
                  use_ohmc = false) -> Float64

Compute target delta. If use_ohmc and ohmc_config set, run OHMC; else BS delta.
"""
function decide_action(
    agent::UnifiedAgent,
    env::TradingEnv,
    agent_state::AgentState;
    use_ohmc = false,
)
    cfg = env.config
    S = env.state.S
    t = env.state.t
    tau = max(0.0, cfg.T * (1 - t / cfg.n_steps))
    # GARCH returns daily variance; annualize for MarketData
    vol_t = sqrt(agent_state.vol_forecast[1] * 252)
    data = Prezo.MarketData(S, cfg.r, vol_t, cfg.q)

    # Compute BS delta first (needed for control variate and fallback)
    opt = agent.option
    bs_delta = if opt isa Prezo.EuropeanCall
        opt_tau = Prezo.EuropeanCall(opt.strike, tau)
        Prezo.greek(opt_tau, Prezo.Delta(), data)
    elseif opt isa Prezo.EuropeanPut
        opt_tau = Prezo.EuropeanPut(opt.strike, tau)
        Prezo.greek(opt_tau, Prezo.Delta(), data)
    else
        opt_tau = typeof(opt)(opt.strike, tau)
        Prezo.numerical_greek(opt_tau, Prezo.Delta(), Prezo.BlackScholes(), data)
    end

    # Option with remaining time to expiry (for OHMC / CVaR-OHMC)
    opt_ohmc = if agent.option isa Prezo.EuropeanCall
        Prezo.EuropeanCall(agent.option.strike, tau)
    else
        Prezo.EuropeanPut(agent.option.strike, tau)
    end

    # Upgrade A: CVaR-OHMC (Lagrangian CVaR inside OHMC) when config set
    if use_ohmc && agent.cvar_ohmc_config !== nothing && agent.option isa Prezo.EuropeanOption && tau > 1e-4
        res_cvar = Prezo.cvar_ohmc_price(opt_ohmc, data, agent.cvar_ohmc_config; rng = cfg.rng)
        hedge_ratios_t0 = @view(res_cvar.hedge_ratios[1, :])
        raw_ohmc_delta = mean(hedge_ratios_t0)
        if !isnan(raw_ohmc_delta) && !isinf(raw_ohmc_delta)
            blended_delta = if agent.use_control_variate
                ohmc_std = std(hedge_ratios_t0)
                adaptive_bs_weight = agent.cv_base_weight +
                    (1 - agent.cv_base_weight) * clamp(ohmc_std * 2, 0.0, 0.5)
                adaptive_bs_weight * bs_delta + (1 - adaptive_bs_weight) * raw_ohmc_delta
            else
                raw_ohmc_delta
            end
            if isnan(agent_state.delta_ema)
                agent_state.delta_ema = blended_delta
            else
                α = 1 - agent.delta_ema_decay
                agent_state.delta_ema = α * blended_delta + agent.delta_ema_decay * agent_state.delta_ema
            end
            smoothed_delta = agent_state.delta_ema
            corrected_delta = apply_gamma_correction(agent, smoothed_delta, S, vol_t, tau, cfg)
            final_delta = apply_belief_consistency(agent, corrected_delta, agent_state)
            return clamp(final_delta, -2.0, 2.0)
        end
    end

    # Standard OHMC when no CVaR-OHMC config
    if use_ohmc && agent.ohmc_config !== nothing && agent.option isa Prezo.EuropeanOption
        # Only run OHMC if there's meaningful time left
        if tau > 1e-4
            res = Prezo.ohmc_price(opt_ohmc, data, agent.ohmc_config; rng = cfg.rng)
            # hedge_ratios[1, :] = hedge at t=0 across all paths
            hedge_ratios_t0 = @view(res.hedge_ratios[1, :])
            raw_ohmc_delta = mean(hedge_ratios_t0)
            
            if !isnan(raw_ohmc_delta) && !isinf(raw_ohmc_delta)
                # Control variate (Upgrade C): blend OHMC with BS delta
                blended_delta = if agent.use_control_variate
                    # Compute OHMC variance to adjust weight
                    ohmc_std = std(hedge_ratios_t0)
                    # Higher variance → more weight on BS delta
                    # Adaptive weight: cv_base_weight + (1 - cv_base_weight) * min(ohmc_std, 0.5)
                    adaptive_bs_weight = agent.cv_base_weight + 
                        (1 - agent.cv_base_weight) * clamp(ohmc_std * 2, 0.0, 0.5)
                    # Blend: w * BS + (1-w) * OHMC
                    adaptive_bs_weight * bs_delta + (1 - adaptive_bs_weight) * raw_ohmc_delta
                else
                    raw_ohmc_delta
                end
                
                # Apply EMA smoothing (Upgrade B variance reduction)
                if isnan(agent_state.delta_ema)
                    # Initialize EMA on first call
                    agent_state.delta_ema = blended_delta
                else
                    # Smooth: delta = α * raw + (1-α) * prev
                    α = 1 - agent.delta_ema_decay
                    agent_state.delta_ema = α * blended_delta + agent.delta_ema_decay * agent_state.delta_ema
                end
                smoothed_delta = agent_state.delta_ema
                # Apply gamma correction (Upgrade D)
                corrected_delta = apply_gamma_correction(agent, smoothed_delta, S, vol_t, tau, cfg)
                # Apply belief consistency (Upgrade F)
                final_delta = apply_belief_consistency(agent, corrected_delta, agent_state)
                return clamp(final_delta, -2.0, 2.0)
            end
        end
        # Fall through to BS delta if OHMC fails or tau too small
    end

    # Fallback to BS delta (already computed above)
    # Apply gamma correction (Upgrade D)
    corrected_delta = apply_gamma_correction(agent, bs_delta, S, vol_t, tau, cfg)
    # Apply belief consistency (Upgrade F)
    return apply_belief_consistency(agent, corrected_delta, agent_state)
end

"""
    apply_gamma_correction(agent, delta, S, vol, tau, cfg) -> Float64

Gamma correction (Upgrade D): h = Δ + k·Γ·S·σ·√Δt

This corrects for discrete-time hedging error, which is dominated by gamma exposure.
The correction term anticipates the delta change from price movement.
"""
function apply_gamma_correction(
    agent::UnifiedAgent,
    delta::Float64,
    S::Float64,
    vol::Float64,  # annualized vol
    tau::Float64,
    cfg::EnvConfig,
)
    k = agent.gamma_correction_k
    k == 0.0 && return delta  # No correction
    
    opt = agent.option
    data = Prezo.MarketData(S, cfg.r, vol, cfg.q)
    
    # Compute gamma
    gamma = if opt isa Prezo.EuropeanCall
        opt_tau = Prezo.EuropeanCall(opt.strike, tau)
        Prezo.greek(opt_tau, Prezo.Gamma(), data)
    elseif opt isa Prezo.EuropeanPut
        opt_tau = Prezo.EuropeanPut(opt.strike, tau)
        Prezo.greek(opt_tau, Prezo.Gamma(), data)
    else
        opt_tau = typeof(opt)(opt.strike, tau)
        Prezo.numerical_greek(opt_tau, Prezo.Gamma(), Prezo.BlackScholes(), data)
    end
    
    # Δt in years (daily = 1/252)
    dt = 1.0 / 252.0
    
    # Correction: k · Γ · S · σ · √Δt
    correction = k * gamma * S * vol * sqrt(dt)
    
    delta + correction
end

"""
    apply_belief_consistency(agent, delta, agent_state) -> Float64

Belief consistency (Upgrade F): scale delta by 1/(1 + λ·CV²)

When the posterior is uncertain (high coefficient of variation CV = std/mean),
reduce the hedge ratio to avoid overtrading on fake precision.
"""
function apply_belief_consistency(
    agent::UnifiedAgent,
    delta::Float64,
    agent_state::AgentState,
)
    λ = agent.belief_penalty_lambda
    λ == 0.0 && return delta  # No penalty
    
    vol_mean = agent_state.vol_forecast[1]
    vol_std = agent_state.posterior_vol_std
    
    # Coefficient of variation (CV)
    vol_mean <= 1e-10 && return delta
    cv = vol_std / vol_mean
    
    # Scale factor: 1/(1 + λ·CV²)
    # When CV is high (uncertain), scale down
    scale = 1.0 / (1.0 + λ * cv^2)
    
    delta * scale
end

# =============================================================================
# Step 5: Score (Huberized loss) — compute tracking error, update EMA
# =============================================================================

"""
    huber_loss(e::Float64, δ::Float64) -> Float64

Huberized quadratic loss (Upgrade I):
    ℓ(e) = e²           if |e| ≤ δ
           2δ|e| - δ²   if |e| > δ

Keeps nice gradients near 0 but doesn't overweight extreme outliers.
When δ = 0, returns pure quadratic loss.
"""
function huber_loss(e::Float64, δ::Float64)
    δ <= 0.0 && return e^2  # Pure quadratic
    abs_e = abs(e)
    if abs_e <= δ
        return e^2
    else
        return 2δ * abs_e - δ^2
    end
end

"""
    score_step!(agent::UnifiedAgent, agent_state::AgentState, obs::Observation,
                prev_portfolio_value::Float64)

Huberized loss (Upgrade I): quadratic for small errors, linear for large.
Updates agent_state.feedback.tracking_loss and agent_state.score_ema.
"""
function score_step!(
    agent::UnifiedAgent,
    agent_state::AgentState,
    obs::Observation,
    prev_portfolio_value::Float64,
)
    # Portfolio value = cash + inventory * S
    portfolio_value = obs.cash + obs.inventory * obs.S
    # Option value = C (mid price)
    option_value = obs.C
    # Tracking error = replication error
    tracking_error = portfolio_value - option_value
    
    # Huberized loss (Upgrade I) - robust to outliers
    loss = huber_loss(tracking_error, agent.huber_delta)

    agent_state.feedback.tracking_loss = loss
    agent_state.feedback.cumulative_loss += loss
    # Update EMA of loss for adaptive policy tuning
    α = 1 - agent.score_ema_decay
    agent_state.score_ema = α * loss + agent.score_ema_decay * agent_state.score_ema

    nothing
end

# =============================================================================
# Step 6: Size position (CVaR) — scale by tail risk
# =============================================================================

"""
    size_by_cvar_forecast(agent::UnifiedAgent, agent_state::AgentState,
                          raw_delta::Float64, S::Float64, n_sim::Int, rng) -> Float64

Simulate potential losses from holding raw_delta shares over horizon, compute CVaR,
and scale position so that CVaR <= L_max.
Returns scaled delta.
"""
function size_by_cvar_forecast(
    agent::UnifiedAgent,
    agent_state::AgentState,
    raw_delta::Float64,
    S::Float64,
    n_sim::Int,
    rng,
)
    agent.L_max == Inf && return raw_delta  # no limit
    abs(raw_delta) < 1e-8 && return raw_delta  # near-zero delta

    # Simulate returns over horizon using vol forecast
    # GARCH returns daily variance; for 1-day returns, use daily vol directly
    daily_vol = sqrt(agent_state.vol_forecast[1])
    # Simulate n_sim 1-step returns (log returns)
    sims = randn(rng, n_sim) .* daily_vol
    
    # Dollar P&L per unit of underlying: S * (exp(r) - 1)
    # Positive when price rises, negative when drops
    unit_pnl = S .* (exp.(sims) .- 1)
    
    # Position P&L = delta * unit_pnl
    # For long delta: gain when price rises, lose when drops
    # For short delta: gain when price drops, lose when rises
    position_pnl = raw_delta .* unit_pnl
    
    # Losses = negative P&L (positive loss = bad)
    losses = -position_pnl
    
    # CVaR of losses per unit of |delta|
    unit_losses = losses ./ abs(raw_delta)
    cvar_unit = compute_cvar(unit_losses, agent.cvar_alpha)
    cvar_unit <= 0 && return raw_delta  # no tail risk (unusual)

    # Max position size such that CVaR(position) <= L_max
    # CVaR(w * L) = w * CVaR(L) for w > 0
    max_delta = agent.L_max / cvar_unit
    
    # Scale down if needed
    if abs(raw_delta) > max_delta
        scale = max_delta / abs(raw_delta)
        agent_state.feedback.constraint_hit = true
        return raw_delta * scale
    end
    
    raw_delta
end

# =============================================================================
# Step 6b: Kelly for scale (Upgrade A Option B)
# =============================================================================

"""
    size_by_kelly_forecast(agent::UnifiedAgent, agent_state::AgentState,
                          raw_delta::Float64, S::Float64, n_sim::Int, rng;
                          risk_free_daily::Float64 = 0.0) -> Float64

Scale position by fractional Kelly using the distribution of one-period hedged returns.

OHMC+CVaR chooses *what* to do (delta); Kelly chooses *how much*.
Uses Prezo.kelly_continuous(μ, σ², r) with simulated returns from vol_forecast.
"""
function size_by_kelly_forecast(
    agent::UnifiedAgent,
    agent_state::AgentState,
    raw_delta::Float64,
    S::Float64,
    n_sim::Int,
    rng;
    risk_free_daily::Float64 = 0.0,
)
    abs(raw_delta) < 1e-8 && return raw_delta

    daily_vol = sqrt(agent_state.vol_forecast[1])
    sims = randn(rng, n_sim) .* daily_vol
    # One-period simple return: (exp(r) - 1)
    returns = exp.(sims) .- 1.0
    μ_return = mean(returns)
    var_return = max(var(returns), 1e-12)

    # Kelly for continuous returns: f* = (μ - r) / σ²
    f_star = Prezo.kelly_continuous(μ_return, var_return, risk_free_daily)
    # Fractional Kelly and clamp to [0, 1] so we don't lever beyond full size
    scale = clamp(agent.kelly_fraction * f_star, 0.0, 1.0)

    raw_delta * scale
end

# =============================================================================
# Step 7: Execute — trade with costs, discreteness, slippage
# =============================================================================

"""
    execute_trade!(env::TradingEnv, target_delta::Float64) -> (shares_traded, cost, fill_price)

Execute hedge to reach target_delta. Returns trade details.
"""
function execute_trade!(env::TradingEnv, target_delta::Float64)
    shares_traded, cost = execute_hedge!(env, target_delta, 1.0)
    fill_price = env.state.S  # simplified; could model slippage
    (shares_traded, cost, fill_price)
end

# =============================================================================
# Event-driven rebalancing (Upgrade E)
# =============================================================================

"""
    should_rebalance(agent::UnifiedAgent, agent_state::AgentState, 
                     target_delta::Float64, current_step::Int) -> Bool

Decide whether to rebalance using threshold policy:
- Rebalance if |target_delta - last_target_delta| > threshold
- OR if days since last trade > max_days_no_trade
- Always trade on first step

This reduces overtrade and transaction costs while bounding tracking error.
"""
function should_rebalance(
    agent::UnifiedAgent,
    agent_state::AgentState,
    target_delta::Float64,
    current_step::Int,
)
    # Always trade on first step
    agent_state.last_trade_step < 0 && return true
    
    # Check delta change threshold
    delta_change = abs(target_delta - agent_state.last_target_delta)
    if delta_change > agent.rebalance_threshold
        return true
    end
    
    # Check max days since last trade
    days_since_trade = current_step - agent_state.last_trade_step
    if days_since_trade >= agent.max_days_no_trade
        return true
    end
    
    false
end

"""
    update_trade_state!(agent_state::AgentState, target_delta::Float64, current_step::Int)

Update agent state after a trade is executed.
"""
function update_trade_state!(agent_state::AgentState, target_delta::Float64, current_step::Int)
    agent_state.last_trade_step = current_step
    agent_state.last_target_delta = target_delta
    nothing
end

# =============================================================================
# Step 8: Feedback — compute realized P&L, drift, update history
# =============================================================================

"""
    update_feedback!(agent_state::AgentState, obs::Observation, prev_obs::Observation,
                     target_delta::Float64, cost::Float64)

Update feedback: realized P&L, inventory drift from target.
"""
function update_feedback!(
    agent_state::AgentState,
    obs::Observation,
    prev_obs::Observation,
    target_delta::Float64,
    cost::Float64,
)
    # Mark-to-market P&L: Δ(cash + inventory*S) - cost
    prev_value = prev_obs.cash + prev_obs.inventory * prev_obs.S
    curr_value = obs.cash + obs.inventory * obs.S
    agent_state.feedback.realized_pnl = curr_value - prev_value

    # Inventory drift: how far actual inventory is from target
    agent_state.feedback.inventory_drift = obs.inventory - target_delta

    nothing
end

"""
    append_return!(agent_state::AgentState, S_prev::Float64, S_curr::Float64; max_len = 100)

Append log return to history (rolling window).
"""
function append_return!(agent_state::AgentState, S_prev::Float64, S_curr::Float64; max_len = 100)
    r = log(S_curr / S_prev)
    push!(agent_state.returns_history, r)
    if length(agent_state.returns_history) > max_len
        popfirst!(agent_state.returns_history)
    end
    nothing
end

# =============================================================================
# Full loop: step_unified! — one iteration of all 8 steps
# =============================================================================

"""
    step_unified!(agent::UnifiedAgent, env::TradingEnv, agent_state::AgentState;
                  use_ohmc = false, cvar_n_sim = 1000) -> Observation

Run one full iteration of the 8-step loop:
1. Observe → 2. Belief update (SMC-ABC) → 3. Forecast (GARCH) → 4. Decide (OHMC/BS)
→ 5. Score (quadratic) → 6. Size (CVaR) → 7. Execute → 8. Feedback

Returns the new Observation after execution.
"""
function step_unified!(
    agent::UnifiedAgent,
    env::TradingEnv,
    agent_state::AgentState;
    use_ohmc = false,
    cvar_n_sim = 1000,
)
    cfg = env.config
    rng = cfg.rng

    # --- Step 1: Observe ---
    # Use agent's vol forecast if available, else NaN (will use true_vol)
    # GARCH returns daily variance; annualize for pricing
    vol_est = length(agent_state.vol_forecast) > 0 ? sqrt(agent_state.vol_forecast[1] * 252) : NaN
    prev_fill = agent_state.feedback.inventory_drift != 0 ? env.state.inventory : 0.0
    prev_obs = observe_full(env, agent.option, prev_fill, env.state.S; vol_est = vol_est)

    # --- Step 2: Belief update (SMC-ABC) ---
    belief_update!(agent, agent_state, rng)

    # --- Step 3: Forecast risk (GARCH) ---
    forecast_risk!(agent, agent_state)

    # --- Step 4: Decide action (OHMC or BS delta) ---
    raw_delta = decide_action(agent, env, agent_state; use_ohmc = use_ohmc)

    # --- Step 5: Score (quadratic loss) — before trade, on previous position ---
    prev_portfolio = prev_obs.cash + prev_obs.inventory * prev_obs.S
    score_step!(agent, agent_state, prev_obs, prev_portfolio)

    # --- Step 6: Size position (CVaR then optionally Kelly) ---
    agent_state.feedback.constraint_hit = false
    sized_delta = size_by_cvar_forecast(agent, agent_state, raw_delta, env.state.S, cvar_n_sim, rng)
    if agent.use_kelly_sizing
        r_daily = env.config.r / 252
        sized_delta = size_by_kelly_forecast(
            agent, agent_state, sized_delta, env.state.S, cvar_n_sim, rng;
            risk_free_daily = r_daily,
        )
    end

    # --- Step 7: Execute ---
    shares_traded, cost, fill_price = execute_trade!(env, sized_delta)

    # --- Step 8: Feedback ---
    # Update vol_est after forecast (annualized for pricing)
    vol_est_post = sqrt(agent_state.vol_forecast[1] * 252)
    new_obs = observe_full(env, agent.option, shares_traded, fill_price; vol_est = vol_est_post)
    update_feedback!(agent_state, new_obs, prev_obs, sized_delta, cost)

    new_obs
end

# =============================================================================
# Legacy API (for backward compatibility with experiments.jl)
# =============================================================================

"""
    observe_unified(agent::UnifiedAgent, env::TradingEnv)

Return simple observation (legacy).
"""
observe_unified(agent::UnifiedAgent, env::TradingEnv) = observe(env)

"""
    update_belief_unified(agent::UnifiedAgent, returns_history::Vector{Float64})

GARCH forecast only (legacy).
"""
function update_belief_unified(agent::UnifiedAgent, returns_history::Vector{Float64})
    if length(returns_history) < 1
        return fill(_garch_unconditional_variance(agent.garch_model), agent.horizon)
    end
    Prezo.forecast(agent.garch_model, returns_history, agent.horizon)
end

"""
    decide_action_unified(agent, env, vol_forecast; use_ohmc) -> (delta,)

Legacy API returning tuple.
"""
function decide_action_unified(
    agent::UnifiedAgent,
    env::TradingEnv,
    vol_forecast::Vector{Float64};
    use_ohmc = false,
)
    # Create temporary agent state with given forecast
    tmp_state = AgentState(agent.horizon)
    tmp_state.vol_forecast .= vol_forecast
    delta = decide_action(agent, env, tmp_state; use_ohmc = use_ohmc)
    (delta,)
end

"""
    assimilate_and_decide(agent, env, returns_history; use_ohmc) -> (delta,)

Legacy one-shot API (Observe + Belief + Forecast + Decide in one call).
"""
function assimilate_and_decide(
    agent::UnifiedAgent,
    env::TradingEnv,
    returns_history::Vector{Float64};
    use_ohmc = false,
)
    rng = env.config.rng
    tmp_state = AgentState(agent.horizon)
    tmp_state.returns_history = copy(returns_history)

    # Belief update (SMC-ABC)
    belief_update!(agent, tmp_state, rng)
    # Forecast (GARCH)
    forecast_risk!(agent, tmp_state)
    # Decide
    delta = decide_action(agent, env, tmp_state; use_ohmc = use_ohmc)
    (delta,)
end
