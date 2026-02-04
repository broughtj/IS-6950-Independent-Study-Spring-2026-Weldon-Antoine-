"""
Order execution with lot rounding, transaction costs, and optional slippage.
"""

"""
    round_to_lot(shares::Real, lot_size::Float64) -> Float64

Round shares to nearest multiple of lot_size.
"""
function round_to_lot(shares::Real, lot_size::Float64)
    lot_size > 0 || return 0.0
    n = round(shares / lot_size)
    return Float64(n * lot_size)
end

"""
    compute_cost(shares_traded::Float64, price::Float64, cost_model::CostModel) -> Float64

Transaction cost = spread component + fixed commission.
Spread is applied as cost on the traded value (half-spread each side approximated as spread_bps * value).
"""
function compute_cost(shares_traded::Float64, price::Float64, cost_model::CostModel)
    value = abs(shares_traded) * price
    spread_cost = value * cost_model.spread_bps
    commission = cost_model.commission_per_trade
    spread_cost + commission
end

"""
    compute_cost_state_dependent(shares_traded, price, cost_model, vol, base_vol;
                                  vol_sensitivity=2.0, impact_coef=0.1) -> Float64

State-dependent transaction costs (Upgrade H):

1. **Vol-dependent spread**: spread widens when vol > base_vol
   spread_effective = spread_base × (1 + vol_sensitivity × max(0, vol/base_vol - 1))

2. **Size-dependent impact**: slippage increases with order size
   impact = impact_coef × |shares| × price × (vol/base_vol)

This makes costs realistic under stress (high vol = wider spreads, larger orders = more impact).
"""
function compute_cost_state_dependent(
    shares_traded::Float64,
    price::Float64,
    cost_model::CostModel,
    vol::Float64,           # Current volatility (annualized)
    base_vol::Float64;      # Base/normal volatility
    vol_sensitivity::Float64 = 2.0,  # How much spread widens with vol
    impact_coef::Float64 = 0.0001,   # Market impact coefficient
)
    abs_shares = abs(shares_traded)
    value = abs_shares * price
    
    # Vol ratio (clamped to avoid extreme values)
    vol_ratio = clamp(vol / max(base_vol, 0.01), 0.5, 5.0)
    
    # 1. Vol-dependent spread widening
    # When vol = base_vol, spread_mult = 1
    # When vol = 2 × base_vol, spread_mult = 1 + vol_sensitivity
    spread_mult = 1.0 + vol_sensitivity * max(0.0, vol_ratio - 1.0)
    spread_cost = value * cost_model.spread_bps * spread_mult
    
    # 2. Size-dependent market impact (square-root model is common, but linear is simpler)
    # Impact scales with size and volatility
    impact_cost = impact_coef * abs_shares * price * vol_ratio
    
    # 3. Fixed commission
    commission = cost_model.commission_per_trade
    
    spread_cost + impact_cost + commission
end

"""
    execute_hedge!(env::TradingEnv, target_delta::Float64, option_contracts::Real = 1.0;
                   lot_size = nothing)

Update hedge: trade shares so that inventory (in shares) equals target_delta * option_contracts.
Applies lot rounding and transaction costs, updates env.state.inventory and env.state.cash.
Returns (shares_traded, cost).
"""
function execute_hedge!(
    env::TradingEnv,
    target_delta::Float64,
    option_contracts::Real = 1.0;
    lot_size = nothing,
)
    cfg = env.config
    lot = lot_size === nothing ? cfg.lot_size : Float64(lot_size)
    target_shares = target_delta * option_contracts
    current_shares = env.state.inventory
    raw_trade = target_shares - current_shares
    shares_traded = round_to_lot(raw_trade, lot)
    price = env.state.S
    cost = compute_cost(shares_traded, price, cfg.cost_model)
    # Update state: buy positive shares (spend cash), sell negative (receive cash)
    env.state = EnvState(
        env.state.S,
        env.state.t,
        current_shares + shares_traded,
        env.state.cash - shares_traded * price - cost,
        env.state.margin_used,
        env.state.path_idx,
    )
    (shares_traded, cost)
end
