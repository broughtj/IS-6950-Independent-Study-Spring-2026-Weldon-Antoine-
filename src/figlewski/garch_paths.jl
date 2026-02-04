"""
GARCH-driven Monte Carlo price paths. Uses Prezo's GJR-GARCH simulate to generate
time-varying volatility and returns, then builds price paths.
"""

"""
    asset_paths_col_garch(S0::Float64, r::Float64, garch_model, T::Float64,
                         n_steps::Int, n_reps::Int; rng = Random.GLOBAL_RNG)

Generate n_reps price paths with GARCH-driven volatility. Each path is built from
GARCH-simulated returns: S[t+1] = S[t] * exp(r_t). Risk-free rate r is not added
to the GARCH returns (GARCH models mean-zero returns); we treat simulated returns
as excess returns and use S[t+1] = S[t] * exp((r - 0.5*σ²)*dt + σ*√dt*Z) with
σ² = h_t from GARCH for consistency with risk-neutral drift. Actually Prezo.simulate
returns raw returns r_t (so log return). So S[t+1] = S[t] * exp(r_t) gives a price
path under the real-world measure. For risk-neutral we could use r_t = (r - 0.5*h_t)*dt + sqrt(h_t*dt)*Z
with dt = T/n_steps. We'll do: per path, simulate (returns, h) from GARCH for n_steps.
Then price path: S[1]=S0, S[t+1]=S[t]*exp(returns[t]). That gives one path. Replicate n_reps times.
Returns matrix (n_steps+1, n_reps).
"""
function asset_paths_col_garch(
    S0::Float64,
    r::Float64,
    garch_model,
    T::Float64,
    n_steps::Int,
    n_reps::Int;
    rng = Random.GLOBAL_RNG,
)
    dt = T / n_steps
    paths = zeros(n_steps + 1, n_reps)
    paths[1, :] .= S0
    for rep in 1:n_reps
        seed = rand(rng, UInt32)
        returns_sim, _ = Prezo.simulate(garch_model, n_steps; seed = seed)
        S = S0
        paths[1, rep] = S
        for t in 1:n_steps
            # Log return from GARCH; price update
            S = S * exp(returns_sim[t])
            paths[t + 1, rep] = S
        end
    end
    paths
end

"""
    asset_paths_col_garch_risk_neutral(S0, r, garch_model, T, n_steps, n_reps; rng)

Generate paths under risk-neutral measure: drift r, volatility from GARCH.
S[t+1] = S[t] * exp((r - 0.5*h_t)*dt + sqrt(h_t*dt)*Z).
"""
function asset_paths_col_garch_risk_neutral(
    S0::Float64,
    r::Float64,
    garch_model,
    T::Float64,
    n_steps::Int,
    n_reps::Int;
    rng = Random.GLOBAL_RNG,
)
    dt = T / n_steps
    paths = zeros(n_steps + 1, n_reps)
    paths[1, :] .= S0
    for rep in 1:n_reps
        seed = rand(rng, UInt32)
        _, h_sim = Prezo.simulate(garch_model, n_steps; seed = seed)
        S = S0
        paths[1, rep] = S
        for t in 1:n_steps
            vol_t = sqrt(max(h_sim[t], 1e-12))
            z = randn(rng)
            S = S * exp((r - 0.5 * vol_t^2) * dt + vol_t * sqrt(dt) * z)
            paths[t + 1, rep] = S
        end
    end
    paths
end
