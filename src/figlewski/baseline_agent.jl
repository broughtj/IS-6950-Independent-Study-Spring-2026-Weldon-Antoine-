"""
Figlewski (1989) baseline agent: fixed volatility estimate, delta hedge, fixed rebalance schedule.
"""

struct FiglewskiAgent{O}
    sigma_hat::Float64
    rebalance_freq::Int
    option::O
end

"""
    observe_baseline(agent::FiglewskiAgent, env::TradingEnv)

Return observation for baseline agent (same as env.observe).
"""
observe_baseline(agent::FiglewskiAgent, env::TradingEnv) = observe(env)

"""
    decide_action_baseline(agent::FiglewskiAgent, env::TradingEnv; r = nothing, q = nothing)

Compute target delta for current state. Returns (delta_target,) for 1 option contract.
Uses Black-Scholes delta with agent.sigma_hat. r, q default to env.config.
"""
function decide_action_baseline(agent::FiglewskiAgent, env::TradingEnv; r = nothing, q = nothing)
    cfg = env.config
    S = env.state.S
    t = env.state.t
    tau = cfg.T * (1 - t / cfg.n_steps)
    tau = max(0.0, tau)
    r_ = r === nothing ? cfg.r : Float64(r)
    q_ = q === nothing ? cfg.q : Float64(q)
    data = Prezo.MarketData(S, r_, agent.sigma_hat, q_)
    # For European options Prezo uses expiry as time to expiry; our option may have been
    # created with full T, so we need market data with spot=S and we need an option
    # with expiry=tau. Prezo.EuropeanCall(strike, expiry) - expiry is time to expiry.
    opt = agent.option
    # If option has fixed expiry, we need delta at current (S, tau). Prezo's greek
    # uses option.expiry. So we must pass market data with current spot and an option
    # with time to expiry = tau. Create a temporary option with same strike, expiry = tau.
    if opt isa Prezo.EuropeanCall
        opt_tau = Prezo.EuropeanCall(opt.strike, tau)
        delta = Prezo.greek(opt_tau, Prezo.Delta(), data)
    elseif opt isa Prezo.EuropeanPut
        opt_tau = Prezo.EuropeanPut(opt.strike, tau)
        delta = Prezo.greek(opt_tau, Prezo.Delta(), data)
    else
        # American or other: use numerical delta with BlackScholes engine
        opt_tau = typeof(opt)(opt.strike, tau)
        engine = Prezo.BlackScholes()
        delta = Prezo.numerical_greek(opt_tau, Prezo.Delta(), engine, data)
    end
    (delta,)
end
