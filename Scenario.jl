
""" simulation result container """
mutable struct Scenario{T}
    T::Int                          # simulation length
    mark::Vector{T}                 # market prices
    oracle::Vector{T}               # oracle prices
    dx::Vector{T}                   # true dx
    net_position::Vector{T}         # net market position: longs - shorts = -dx
    C::Vector{T}                    # price peg C
    k::Vector{T}                    # liquidity depths k
    x::Vector{T}                    # base asset reserves
    y::Vector{T}                    # quote asset reserves
    funding::Vector{T}              # funding costs
    op_cost::Vector{T}              # operating cost of changing C and k
    revenue::Vector{T}              # net revenue: -funding cost - operating cost
    unit_slippage::Vector{T}        # slippage of unit trade: 1/(x+1)
    divergence::Vector{T}           # (mark-oracle)/oracle
    forecasts::Vector{Array{T}}     # forecasts of true values used to make decisions
    function Scenario(; T = Float64)
        new{T}(0, [T[] for i = 1:14]...)
    end
end

""" update simulation result """
function update!(f::Scenario, pₒ, x, C, k, forecast)
    if length(f.x) == 0
        dx = 0
        op_cost = 0
    else
        dx = x - f.x[end]
        ΔC = C - f.C[end]
        Δk = k - f.k[end]
        Δx = -dx
        op_cost = operating_cost(f.C[end], f.k[end], f.x[end], ΔC, Δk, Δx)
    end
    pₘ = C * k / x^2
    funding = funding_cost(pₘ, pₒ, -dx)

    f.T += 1
    push!(f.mark, pₘ)
    push!(f.oracle, pₒ)
    push!(f.dx, dx)
    push!(f.net_position, -dx)
    push!(f.C, C)
    push!(f.k, k)
    push!(f.x, x)
    push!(f.y, k / x)
    push!(f.funding, funding)
    push!(f.op_cost, op_cost)
    push!(f.revenue, -funding - op_cost)
    push!(f.unit_slippage, 1 / (x + 1))
    push!(f.divergence, (pₘ - pₒ) / pₒ)
    push!(f.forecasts, forecast)
end

""" cost of changing C and k """
function operating_cost(C, k, x, ΔC, Δk, Δx)
    return (ΔC * k * Δx / x - ΔC * Δk - C * Δk) / (x + Δx)
end

""" cost of funding longs/shorts """
function funding_cost(pₘ, pₒ, net_position)
    return -net_position * (pₘ - pₒ) / 24
end
