using LinearAlgebra, Statistics, Convex, ECOS
import Plots
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)

include("Scenario.jl")
include("Simulator.jl")
include("MPC.jl")

# time horizon and initial market state
T = 500
x0 = 300
p0 = 100

# market behavior
σ_oracle = 1.0
ϵ_oracle = 1e-1
μ_dx = 5.0
σ_dx = 1e-1

# policy parameters
forecast_length = 20
base_diag_cost = [1; 1e-8; 1e-4; 1e-1]
lookahead = 15
λ_k = 1

policies = [
    ("do_nothing", ad_hoc_policy(false, false)),
    ("adhoc_adjust_C", ad_hoc_policy(true, false)),
    ("adhoc_adjust_k", ad_hoc_policy(false, true)),
    ("adhoc_adjust_both", ad_hoc_policy(true, true)),
    (
        "mpc_policy",
        make_mpc_policy(base_diag_cost, lookahead, λ_k; solver = ECOS.Optimizer),
    ),
    (
        "mpc_price_tracker",
        make_mpc_policy(base_diag_cost, lookahead, λ_k; use_offdiagonal_costs = false, solver = ECOS.Optimizer),
    ),
]

""" visually check policy performance """
function visual_eval()
    for idx = 1:length(policies)
        name, policy = policies[idx]
        f = PerpSimulator(σ_oracle, ϵ_oracle, μ_dx, σ_dx)
        po = sample_oracle_price(f, p0, T)

        res = simulate(f, po, x0; policy = policy, forecast_length = forecast_length)

        plot(res.mark, label = "mark")
        plot!(res.oracle, label = "oracle")
        xlabel!("Time")
        ylabel!("Price")
        savefig("assets/$(name)_mark_vs_oracle.png")

        plot(res.net_position, label = "net position (long - short)")
        plot!(res.divergence * 10, label = "scaled and signed price divergence")
        savefig("assets/$(name)_position_vs_divergence.png")

        plot(res.revenue, label = "revenue")
        savefig("assets/$(name)_funding_revenue.png")

        net_rev = sum(res.revenue)
        mse_div = mean((res.mark .- res.oracle) .^ 2)
        mean_slippage = mean(res.unit_slippage)
        println(
            "net revenue = $(net_rev), funding rev = $(-sum(res.funding)), mse div = $(mse_div), mean slippage = $(mean_slippage)",
        )
    end
end

""" visually check forecast quality """
function check_forecasts()
    t = rand(1:T-forecast_length-1)
    labels = ["dx̂", "x̂", "p̂ₘ", "p̂ₒ"]
    sim = [
        res.dx[t:t+forecast_length],
        res.x[t:t+forecast_length],
        res.mark[t:t+forecast_length],
        res.oracle[t:t+forecast_length],
    ]

    for (i, l) in enumerate(labels)
        plot(res.forecasts[t][:, i], label = "$(l) pred")
        plot!(sim[i], label = "$(l) sim")
        savefig("assets/forecast_$(l).png")
    end
end

""" Evaluate average performance of different policies """
function evaluate(policies; num_oracle_trajectories = 10)
    # compute metrics given simulation result
    function _metrics(res)
        rev = sum(res.revenue)
        mean_abs_div = mean(abs.((res.mark .- res.oracle) ./ res.oracle))
        mean_slippage = mean(res.unit_slippage)
        return [rev, mean_abs_div, mean_slippage]
    end

    f = PerpSimulator(σ_oracle, ϵ_oracle, μ_dx, σ_dx)
    metrics = Dict()
    for (name, _) in policies
        metrics[name] = []
    end

    for _ = 1:num_oracle_trajectories
        po = sample_oracle_price(f, p0, T)
        for (name, policy) in policies
            res = simulate(f, po, x0; policy = policy, forecast_length = forecast_length)
            push!(metrics[name], _metrics(res))
        end
    end

    means = Dict()
    for (name, _) in policies
        values = hcat(metrics[name]...)'
        means[name] = mean(values, dims = 1)
    end

    return metrics, means
end
