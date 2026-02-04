"""
Performance metrics and comparison for Figlewski vs unified agent.
"""

struct ExperimentResult
    mean_pnl::Float64
    std_pnl::Float64
    cvar_loss::Float64
    tracking_error_mean::Float64
    tracking_error_std::Float64
    turnover::Float64
    trade_count::Int
    max_drawdown::Float64
    constraint_hits::Int
    arbitrage_band_width::Float64
    n_paths::Int
    # Figlewski (1989) specific metrics
    residual_vol_pct::Float64      # std(pnl) / option_value * 100
    shares_per_share_hedged::Float64  # total shares traded / (n_paths * delta_avg)
end

function ExperimentResult(
    pnls::AbstractVector{<:Real},
    tracking_errors::AbstractVector{<:Real},
    turnover::Real,
    trade_count::Int,
    constraint_hits::Int = 0;
    alpha_cvar = 0.95,
    arbitrage_band_width = NaN,
    option_value = NaN,  # For required spread metric
    underlying_value = NaN,  # S0 for Figlewski residual vol %
    total_shares_traded = NaN,  # For shares per share hedged
)
    n = length(pnls)
    n > 0 || return ExperimentResult(
        0.0, 0.0, 0.0, 0.0, 0.0, Float64(turnover), trade_count, 0.0, constraint_hits,
        isnan(arbitrage_band_width) ? 0.0 : arbitrage_band_width, 0, 0.0, 0.0,
    )
    underlying_value = isnan(underlying_value) ? 100.0 : underlying_value  # default to 100
    mean_pnl = mean(pnls)
    std_pnl = std(pnls)
    sorted = sort(pnls)
    # CVaR of loss = expected loss given loss >= VaR
    idx = max(1, floor(Int, (1 - alpha_cvar) * n))
    tail = @view sorted[1:idx]
    cvar_loss = mean(tail)
    te_mean = length(tracking_errors) > 0 ? mean(tracking_errors) : 0.0
    te_std = length(tracking_errors) > 0 ? std(tracking_errors) : 0.0
    # Max drawdown from cumulative PnL
    cum = 0.0
    peak = 0.0
    dd = 0.0
    for x in pnls
        cum += x
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    end
    # Figlewski metrics
    # Residual vol % = std(P&L) / S0 * 100 (as in Figlewski 1989)
    residual_vol_pct = isnan(underlying_value) || underlying_value <= 0 ? 0.0 : (std_pnl / underlying_value) * 100
    shares_per_share = isnan(total_shares_traded) ? 0.0 : total_shares_traded / max(1, n)
    ExperimentResult(
        mean_pnl, std_pnl, cvar_loss, te_mean, te_std,
        Float64(turnover), trade_count, dd, constraint_hits,
        isnan(arbitrage_band_width) ? 0.0 : arbitrage_band_width, n,
        residual_vol_pct, shares_per_share,
    )
end

function Base.show(io::IO, r::ExperimentResult)
    print(io, "ExperimentResult(mean_pnl=", round(r.mean_pnl; digits=4), ", std_pnl=",
          round(r.std_pnl; digits=4), ", cvar_loss=", round(r.cvar_loss; digits=4),
          ", turnover=", round(r.turnover; digits=2), ", n_paths=", r.n_paths)
    if r.residual_vol_pct > 0
        print(io, ", residual_vol%=", round(r.residual_vol_pct; digits=2))
    end
    print(io, ")")
end

"""
    compare_agents(result_A::ExperimentResult, result_B::ExperimentResult;
                   label_A = "A", label_B = "B")

Return a named tuple of relative differences and a short summary.
"""
function compare_agents(
    result_A::ExperimentResult,
    result_B::ExperimentResult;
    label_A = "A",
    label_B = "B",
)
    diff_mean_pnl = result_B.mean_pnl - result_A.mean_pnl
    diff_std_pnl = result_B.std_pnl - result_A.std_pnl
    diff_cvar = result_B.cvar_loss - result_A.cvar_loss
    diff_te_mean = result_B.tracking_error_mean - result_A.tracking_error_mean
    diff_turnover = result_B.turnover - result_A.turnover
    (
        label_A = label_A,
        label_B = label_B,
        diff_mean_pnl = diff_mean_pnl,
        diff_std_pnl = diff_std_pnl,
        diff_cvar_loss = diff_cvar,
        diff_tracking_error_mean = diff_te_mean,
        diff_turnover = diff_turnover,
        result_A = result_A,
        result_B = result_B,
    )
end
