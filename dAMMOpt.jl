using LinearAlgebra, Statistics, Convex, ECOS, Plots

include("Scenario.jl")
include("Simulator.jl")



f = PerpSimulator(1.0, 1e-1, 5.0, 1e-1)
po = sample_oracle_price(f, 100, 1000)
res = simulate(f, po, 300; forecast_length=5)

plot(res.mark, label = "mark")
plot!(res.oracle, label = "oracle")
savefig("assets/mark_vs_oracle.png")

plot(res.net_position, label = "net position (long - short)")
plot!(res.divergence * 10, label = "scaled and signed price divergence")
savefig("assets/position_vs_divergence.png")

plot(res.revenue, label = "funding revenue")
savefig("assets/funding_revenue.png")
