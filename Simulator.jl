
""" perpetual swap simulator with unbiased forecasts """
struct PerpSimulator{T}
    σ_oracle::T     # std. of true oracle random walk
    ϵ_oracle::T     # std. of oracle forecast
    μ_dx::T         # mean of trade activity dx
    σ_dx::T         # std. of trade activity dx
    function PerpSimulator(σ_oracle::T, ϵ_oracle::T, μ_dx::T, σ_dx::T) where {T}
        new{T}(σ_oracle, ϵ_oracle, μ_dx, σ_dx)
    end
end

""" simulate market for a fixed oracle price sequence """
function simulate(
    f::PerpSimulator,
    pₒ::Vector,
    x0::Real;
    policy = (res, forecast, est_net_position) -> (0, 0),
    forecast_length::Int = 1,
)::Scenario
    res = Scenario()
    for t = 1:length(pₒ)-forecast_length
        # estimate oracle price
        p̂ₒ = estimate_oracle_price(f, pₒ[t:end], forecast_length)

        if t == 1
            # initialize market so that:
            # 1. x(1) = y(1) = √k(1)
            # 2. the initial market price equals the initial oracle price: Cy/x = C = pₒ
            x = x0
            C = pₒ[1]
            k = x0^2

            # make forecasts and evaluate policy
            # estimate of market price: assume that C and k are fixed
            dx̂ = vcat(estimate_dx(f, pₒ[1], pₒ[1]), zeros(forecast_length - 1))
            x̂ = vcat(x + dx̂[1], zeros(forecast_length - 1))
            p̂ₘ = vcat(C * k / x̂[1]^2, zeros(forecast_length - 1))
            for i = 2:forecast_length
                p̂ₘ[i] = C * k / x̂[i-1]^2
                dx̂[i] = estimate_dx(f, p̂ₘ[i], p̂ₒ[i])
                x̂[i] = x̂[i-1] + dx̂[i]
            end
            est_net_position = -dx̂
            forecast = [dx̂;; x̂;; p̂ₘ;; p̂ₒ]
        else
            # make forecasts and evaluate policy
            # estimate of market price: assume that C and k are fixed
            dx̂ = vcat(
                estimate_dx(f, res.mark[end], res.oracle[end]),
                zeros(forecast_length - 1),
            )
            x̂ = vcat(res.x[end] + dx̂[1], zeros(forecast_length - 1))
            p̂ₘ = vcat(res.C[end] * res.k[end] / x̂[1]^2, zeros(forecast_length - 1))
            for i = 2:forecast_length
                p̂ₘ[i] = res.C[end] * res.k[end] / x̂[i-1]^2
                dx̂[i] = estimate_dx(f, p̂ₘ[i], p̂ₒ[i])
                x̂[i] = x̂[i-1] + dx̂[i]
            end
            est_net_position = -dx̂
            forecast = [dx̂;; x̂;; p̂ₘ;; p̂ₒ]

            # evaluate policy
            ΔC, Δk = policy(res, forecast, est_net_position)

            # propagate market
            x = res.x[end] + sample_dx(f, res.mark[end], res.oracle[end])
            C = res.C[end] + ΔC
            k = res.k[end] + Δk
        end
        update!(res, pₒ[t], x, C, k, forecast)
    end
    return res
end

""" Use proposed heuristics from Drift team on formulaic parameter adjustment"""
function ad_hoc_policy(adjust_C, adjust_k)
    function _ad_hoc_policy(res::Scenario, forecast::Matrix, est_net_position::Vector)
        C, k, pₘ, pₒ, x, net_pos = res.C[end],
        res.k[end],
        res.mark[end],
        res.oracle[end],
        res.x[end],
        res.net_position[end]
        LIQ_LIMIT = 0.05

        # 1) formulaic peg
        # Cₒ = pₒ * x^2 / k
        Cₒ = forecast[1, 4] * x^2 / k
        ΔC = Cₒ - C
        # ΔC = 0
        d = (pₘ - pₒ) / pₒ
        t = length(res.C)

        repeg_cost = operating_cost(C, k, x, ΔC, 0, net_pos)

        if adjust_C && repeg_cost > -20
            # a) repeg towards (to) oracle if the divergence is too large
            if d > LIQ_LIMIT
                # println("hit liq limit at t = $(t), repegging")
                return (ΔC, 0)
            end

            # b) if the predicted funding payments are large (fee pool drained), let's repeg now
            pred_funding_cost = sum(
                funding_cost(p̂ₘ, p̂ₒ, est_net_pos) for (p̂ₘ, p̂ₒ, est_net_pos) in
                zip(forecast[:, 3], forecast[:, 4], est_net_position)
            )

            # println("pred funding cost = $(pred_funding_cost), repeg cost = $(repeg_cost)")
            if repeg_cost < pred_funding_cost
                # println("repeg cost < funding cost $(t), repegging")
                return (ΔC, 0)
            end
        end

        # 2) formulaic k
        H = 10
        α = 5
        # check for persistent k-bias (cumulative error over the past H timesteps)
        if t > H
            k_bias = sum(
                (res.mark[end-i] .- res.oracle[end-i]) * (res.net_position[end-i] > 0)
                for i = 1:H
            )
            imbalance = sum((res.net_position[end-i] > 0) for i = 0:H-1)
            # println("k_bias = $(k_bias), imbalance = $(imbalance)/$(H)")

            if abs(k_bias) > 2 && imbalance / H > 0.7
                # if longs > shorts
                # a) if mark < oracle, k was too large (decrease)
                # b) if mark > oracle, k was too small (increase)
                Δk = α * k_bias
                # println("Δk = $(Δk)")
                if adjust_k
                    return (0, Δk)
                end
            end
        end
        return (0, 0)
    end
    return _ad_hoc_policy
end



""" Sample true oracle price as a random walk """
function sample_oracle_price(f::PerpSimulator, p₀::Real, T::Int)
    p = zeros(Float64, T)
    for t = 1:T
        p[t] = (t == 1 ? p₀ : p[t-1] + f.σ_oracle * randn())
    end
    return p
end

""" Estimate the oracle price """
function estimate_oracle_price(f::PerpSimulator, pₒ::Vector, lookahead::Int)
    return pₒ[1:lookahead] + f.ϵ_oracle * randn(lookahead)
end

""" Sample true trade activity dx """
function sample_dx(f::PerpSimulator, pₘ::Real, pₒ::Real)
    # if oracle > mark  and shorts > longs, shorts pay longs, perp collects excess
    # would expect shorts to reverse their position (buy)
    # therefore, increase in buying activity means dx should decrease,
    # and so dx is correlated with mark - oracle
    return f.μ_dx * (pₘ - pₒ) / pₒ + f.σ_dx * randn()
end

""" Estimate trade activity dx """
function estimate_dx(f::PerpSimulator, pₘ::Real, pₒ::Real)
    return f.μ_dx * (pₘ - pₒ) / pₒ
end
