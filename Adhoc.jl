
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
        t = length(res.C)

        # 2) formulaic k
        H = 10
        α = 45
        Δk = 0
        # check for persistent k-bias (cumulative error over the past H timesteps)
        if t > H && adjust_k
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
                Δk = -α * k_bias
                # println("Δk = $(Δk)")
                # if adjust_k
                #     return (0, Δk)
                # end
            end
        end

        # 1) formulaic peg
        # Cₒ = pₒ * x^2 / k
        Cₒ = forecast[1, 4] * x^2 / k
        ΔC = Cₒ - C
        # ΔC = 0
        d = (pₘ - pₒ) / pₒ

        repeg_cost = operating_cost(C, k, x, ΔC, 0, net_pos)

        if adjust_C && repeg_cost > -20
            # a) repeg towards (to) oracle if the divergence is too large
            if d > LIQ_LIMIT
                # println("hit liq limit at t = $(t), repegging")
                return (ΔC, Δk)
            end

            # b) if the predicted funding payments are large (fee pool drained), let's repeg now
            pred_funding_cost = sum(
                funding_cost(p̂ₘ, p̂ₒ, est_net_pos) for (p̂ₘ, p̂ₒ, est_net_pos) in
                zip(forecast[:, 3], forecast[:, 4], est_net_position)
            )

            # println("pred funding cost = $(pred_funding_cost), repeg cost = $(repeg_cost)")
            if repeg_cost < pred_funding_cost
                # println("repeg cost < funding cost $(t), repegging")
                return (ΔC, Δk)
            end
        end
        return (0, Δk)
    end
    return _ad_hoc_policy
end
