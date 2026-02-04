"""
Trading environment shared by Figlewski baseline and unified agent.
State, config, GBM price evolution, observation, and step.
"""

# -----------------------------------------------------------------------------
# Cost model: proportional spread + fixed commission
# -----------------------------------------------------------------------------
struct CostModel
    spread_bps::Float64   # bid-ask as fraction of price (e.g. 0.001 = 10 bps)
    commission_per_trade::Float64
end

CostModel(; spread_bps = 0.001, commission_per_trade = 0.0) =
    CostModel(spread_bps, commission_per_trade)

# -----------------------------------------------------------------------------
# Environment state
# -----------------------------------------------------------------------------
struct EnvState
    S::Float64           # current spot
    t::Int               # current step index (0 to n_steps)
    inventory::Float64   # shares held (can be fractional for simulation)
    cash::Float64        # cash balance
    margin_used::Float64 # optional margin (0 if not used)
    path_idx::Int        # which MC path we're on (for path-dependent env)
end

# -----------------------------------------------------------------------------
# Environment config
# -----------------------------------------------------------------------------
struct EnvConfig
    S0::Float64
    r::Float64
    q::Float64
    T::Float64
    n_steps::Int
    true_vol::Float64
    cost_model::CostModel
    lot_size::Float64
    position_limit::Float64
    rng::AbstractRNG
end

function EnvConfig(;
    S0 = 100.0,
    r = 0.05,
    q = 0.0,
    T = 1.0,
    n_steps = 252,
    true_vol = 0.2,
    cost_model = CostModel(),
    lot_size = 1.0,
    position_limit = Inf,
    rng = Random.GLOBAL_RNG,
)
    EnvConfig(S0, r, q, T, n_steps, true_vol, cost_model, lot_size, position_limit, rng)
end

# -----------------------------------------------------------------------------
# Trading environment
# -----------------------------------------------------------------------------
mutable struct TradingEnv
    config::EnvConfig
    state::EnvState
    # Pre-generated price path for this run (column = one path; we use one column at a time)
    price_path::Vector{Float64}  # length n_steps+1, current path
    current_path_idx::Int
    all_paths::Union{Matrix{Float64}, Nothing}  # (n_steps+1, n_paths) if pre-generated
end

"""
    TradingEnv(config; n_paths = 1)

Build environment. If n_paths > 1, can run multiple paths; paths are generated on demand
unless you pass precomputed paths.
"""
function TradingEnv(config::EnvConfig; n_paths::Int = 1, precomputed_paths = nothing)
    path = if precomputed_paths !== nothing
        precomputed_paths[:, 1]
    else
        # Single path placeholder; will be filled by reset!
        zeros(config.n_steps + 1)
    end
    state = EnvState(config.S0, 0, 0.0, 0.0, 0.0, 1)
    TradingEnv(config, state, path, 1, precomputed_paths)
end

"""
    observe(env::TradingEnv)

Return current observation: (S_t, time_to_expiry, step_index).
"""
function observe(env::TradingEnv)
    s = env.state
    cfg = env.config
    tau = max(0.0, cfg.T * (1 - s.t / cfg.n_steps))
    (S_t = s.S, time_to_expiry = tau, step = s.t, inventory = s.inventory, cash = s.cash)
end

"""
    reset!(env::TradingEnv, path_idx = 1)

Reset environment to t=0 and optionally select path index.
If env has precomputed paths, use path_idx; else simulate one GBM path.
"""
function reset!(env::TradingEnv, path_idx::Int = 1)
    cfg = env.config
    if env.all_paths !== nothing && 1 <= path_idx <= size(env.all_paths, 2)
        env.price_path .= env.all_paths[:, path_idx]
        env.current_path_idx = path_idx
    else
        # Generate single GBM path
        dt = cfg.T / cfg.n_steps
        nudt = (cfg.r - cfg.q - 0.5 * cfg.true_vol^2) * dt
        sidt = cfg.true_vol * sqrt(dt)
        env.price_path[1] = cfg.S0
        for j in 2:(cfg.n_steps + 1)
            env.price_path[j] = env.price_path[j-1] * exp(nudt + sidt * randn(cfg.rng))
        end
        env.current_path_idx = 1
    end
    env.state = EnvState(
        env.price_path[1],
        0,
        0.0,
        0.0,
        0.0,
        path_idx,
    )
    observe(env)
end

"""
    step!(env::TradingEnv, action; option = nothing, engine = nothing, market_data = nothing)

Advance one step. Action is (delta_target, shares_to_trade) or just delta_target.
If option and engine and market_data are provided, we can compute option value for reward.
Returns (next_obs, reward, info).
"""
function step!(
    env::TradingEnv,
    action;
    option = nothing,
    engine = nothing,
    market_data = nothing,
)
    cfg = env.config
    s = env.state
    t = s.t
    if t >= cfg.n_steps
        return observe(env), 0.0, (done = true,)
    end
    # Advance price
    next_S = env.price_path[t + 2]
    next_t = t + 1
    # Parse action: can be (delta_target,) or (delta_target, size) or just number
    delta_target = action isa Tuple ? action[1] : Float64(action)
    # Execution is handled by caller (execute_hedge!) or we do minimal update
    # Here we just update state to next period; trade execution applied separately
    new_inventory = s.inventory
    new_cash = s.cash
    reward = 0.0
    if option !== nothing && engine !== nothing && market_data !== nothing
        # Placeholder: reward could be negative squared replication error
        reward = 0.0
    end
    env.state = EnvState(next_S, next_t, new_inventory, new_cash, s.margin_used, s.path_idx)
    info = (done = next_t >= cfg.n_steps, step = next_t, S = next_S)
    (observe(env), reward, info)
end

"""
    generate_paths(env_config::EnvConfig, n_paths::Int)

Generate n_paths GBM paths for the environment using Prezo.
Returns matrix (n_steps+1, n_paths).
"""
function generate_paths(env_config::EnvConfig, n_paths::Int)
    engine = Prezo.MonteCarlo(env_config.n_steps, n_paths)
    Prezo.asset_paths_col(
        engine,
        env_config.S0,
        env_config.r - env_config.q,
        env_config.true_vol,
        env_config.T,
    )
end
