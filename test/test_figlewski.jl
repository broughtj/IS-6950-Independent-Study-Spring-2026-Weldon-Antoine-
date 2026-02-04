using Praxeology
using Praxeology.Figlewski
using Prezo
using Random
using Test

@testset "Figlewski environment" begin
    cfg = Figlewski.EnvConfig(S0 = 100.0, n_steps = 21, true_vol = 0.2)
    env = Figlewski.TradingEnv(cfg)
    obs = Figlewski.reset!(env)
    @test obs.S_t == 100.0
    @test obs.step == 0
    paths = Figlewski.generate_paths(cfg, 5)
    @test size(paths) == (22, 5)
end

@testset "Figlewski execution" begin
    @test Figlewski.round_to_lot(3.7, 1.0) == 4.0
    @test Figlewski.round_to_lot(-2.3, 1.0) == -2.0
    cm = Figlewski.CostModel(0.001, 1.0)
    @test Figlewski.compute_cost(10.0, 100.0, cm) == 10.0 * 100.0 * 0.001 + 1.0
end

@testset "Figlewski CVaR" begin
    losses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # α=0.9: worst 10% tail = [10], mean 10
    @test Figlewski.compute_cvar(losses, 0.9) ≈ 10.0
    w = Figlewski.size_by_cvar(losses, 5.0, 0.9)
    @test w ≈ 0.5  # L_max/CVaR = 5/10
end

@testset "Figlewski metrics" begin
    r = Figlewski.ExperimentResult([1.0, 2.0, 3.0], [0.1, 0.2], 100.0, 50)
    @test r.mean_pnl == 2.0
    @test r.n_paths == 3
    r2 = Figlewski.ExperimentResult([0.5, 1.5], [0.05], 80.0, 40)
    comp = Figlewski.compare_agents(r, r2; label_A = "A", label_B = "B")
    # diff_mean_pnl = result_B - result_A = mean([0.5,1.5]) - mean([1,2,3]) = 1 - 2 = -1
    @test comp.diff_mean_pnl ≈ -1.0
end

@testset "Figlewski baseline agent" begin
    opt = Prezo.EuropeanCall(100.0, 1.0)
    agent = Figlewski.FiglewskiAgent(0.2, 1, opt)
    cfg = Figlewski.EnvConfig(S0 = 100.0, n_steps = 10, true_vol = 0.2)
    env = Figlewski.TradingEnv(cfg)
    Figlewski.reset!(env)
    delta = Figlewski.decide_action_baseline(agent, env)[1]
    @test 0 <= delta <= 1
end

@testset "Figlewski GARCH paths" begin
    garch = Prezo.GJRGARCH(1e-5, 0.05, 0.1, 0.85)
    paths = Figlewski.asset_paths_col_garch(100.0, 0.05, garch, 1.0, 20, 5; rng = MersenneTwister(1))
    @test size(paths) == (21, 5)
    @test paths[1, :] == fill(100.0, 5)
end

@testset "Figlewski run_experiment" begin
    rng = Random.MersenneTwister(123)
    cfg = Figlewski.EnvConfig(S0 = 100.0, n_steps = 21, true_vol = 0.2, rng = rng)
    opt = Prezo.EuropeanCall(100.0, 1.0)
    baseline = Figlewski.FiglewskiAgent(0.2, 1, opt)
    result = Figlewski.run_experiment(baseline, cfg, 20)
    @test result.n_paths == 20
    @test result.trade_count >= 0
end
