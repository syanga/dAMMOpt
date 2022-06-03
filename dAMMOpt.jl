using LinearAlgebra, Statistics, Convex, ECOS, Plots

include("Scenario.jl")
include("Simulator.jl")

T = 1000
forecast_length = 5
f = PerpSimulator(1.0, 1e-1, 5.0, 1e-1)
po = sample_oracle_price(f, 100, T)
res = simulate(f, po, 300; policy=ad_hoc_policy(false, false), forecast_length=forecast_length)

plot(res.mark, label = "mark")
plot!(res.oracle, label = "oracle")
savefig("assets/mark_vs_oracle.png")

plot(res.net_position, label = "net position (long - short)")
plot!(res.divergence * 10, label = "scaled and signed price divergence")
savefig("assets/position_vs_divergence.png")

plot(res.revenue, label = "revenue")
savefig("assets/funding_revenue.png")

net_rev = sum(res.revenue)
mse_div = mean((res.mark .- res.oracle).^2)
mean_slippage = mean(res.unit_slippage)
println("net revenue = $(net_rev), funding rev = $(-sum(res.funding)), mse div = $(mse_div), mean slippage = $(mean_slippage)")

# check forecast vs. sim (truth)
t = rand(1:T-forecast_length-1)
labels = ["dx̂", "x̂", "p̂ₘ", "p̂ₒ"]
sim = [
  res.dx[t:t+forecast_length], 
  res.x[t:t+forecast_length], 
  res.mark[t:t+forecast_length], 
  res.oracle[t:t+forecast_length]
]

for (i,l) = enumerate(labels)
  plot(res.forecasts[t][:, i], label="$(l) pred")
  plot!(sim[i], label="$(l) sim")
  savefig("assets/forecast_$(l).png")
end

policies = [
  ("do_nothing", ad_hoc_policy(false, false)),
  ("adhoc_adjust_C", ad_hoc_policy(true, false)),
  ("adhoc_adjust_k", ad_hoc_policy(false, true)),
  ("adhoc_adjust_both", ad_hoc_policy(true, true))
]

function evaluate(policies; num_oracle_trajectories=1000)
  T = 1000
  forecast_length = 5
  f = PerpSimulator(1.0, 1e-1, 5.0, 1e-1)
  metrics = Dict()
  for (name, _) = policies
    metrics[name] = []
  end

  for _ = 1:num_oracle_trajectories
    po = sample_oracle_price(f, 100, T)
    for (name, policy) = policies
      res = simulate(f, po, 300; policy=policy, forecast_length=forecast_length)
      push!(metrics[name], _metrics(res))
    end
  end
  
  means = Dict()
  for (name, _) = policies
    values = hcat(metrics[name]...)'
    means[name] = mean(values, dims=1)
  end

  return metrics, means
end

function _metrics(res)
  rev = sum(res.revenue)
  mean_abs_div = mean(abs.((res.mark .- res.oracle)./res.oracle))
  mean_slippage = mean(res.unit_slippage)
  return [rev, mean_abs_div, mean_slippage]
end

