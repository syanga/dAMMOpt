using LinearAlgebra, Statistics, Convex, ECOS, Plots
Plots.scalefontsizes()
Plots.scalefontsizes(1.5)

include("Scenario.jl")
include("Simulator.jl")
include("Adhoc.jl")
include("MPC.jl")
include("Evaluation.jl")

# time horizon and initial market state
T = 500
p0 = 100
x0 = 300

# market behavior
σ_oracle = 1.0
ϵ_oracle = 1e-1
μ_dx = 5.0
σ_dx = 1e-1
f = PerpSimulator(σ_oracle, ϵ_oracle, μ_dx, σ_dx)
po = sample_oracle_price(f, p0, T)

# policy parameters
forecast_length = 20
base_diag_cost = [1; 1e-6; 1e-4; 0.05]
lookahead = 15
λ_k = 100

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

# generate plots
visual_eval(f, policies, po, x0)

# evaluate policies on several metrics
metrics, means = evaluate_policies(f, policies, x0; num_oracle_trajectories = 1)
