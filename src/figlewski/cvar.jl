"""
CVaR (Conditional Value at Risk) computation and position sizing.
"""

"""
    compute_cvar(losses::Vector{Float64}, alpha::Float64) -> Float64

CVaR at level α: expected loss given loss >= VaR_α.
losses are positive = loss. α = 0.95 means we look at the worst 5% tail.
"""
function compute_cvar(losses::Vector{Float64}, alpha::Float64)
    isempty(losses) && return 0.0
    sorted = sort(losses)
    n = length(sorted)
    # Worst (1-α) fraction: largest losses
    k = max(1, floor(Int, (1 - alpha) * n))
    mean(@view(sorted[n-k+1:n]))
end

"""
    size_by_cvar(unit_losses::Vector{Float64}, L_max::Float64, alpha::Float64) -> Float64

Max position size such that CVaR_α(scaled losses) <= L_max.
With linear scaling, CVaR(w * L) = w * CVaR(L), so w_max = L_max / CVaR_α(unit_losses).
"""
function size_by_cvar(unit_losses::Vector{Float64}, L_max::Float64, alpha::Float64)
    L_max <= 0 && return 0.0
    c = compute_cvar(unit_losses, alpha)
    c <= 0 && return Inf
    L_max / c
end
