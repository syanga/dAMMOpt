
""" MPC policy """
function make_price_tracker_policy(base_diag_cost, lookahead, λ_k; solver=ECOS.Optimizer)
    @assert lookahead > 0   
    @assert length(base_diag_cost) == 4
    function mpc_policy(res, forecast, est_net_position)
        Cₜ, kₜ, p̂ₒ, x̂ = res.C[end], res.k[end], forecast[:, 4], forecast[:, 2]
        # nominal trajectories k̂, Ĉ
        k̂ = [kₜ + λ_k * τ for τ = 0:lookahead]
        Ĉ = vcat(Cₜ, [p̂ₒ[τ] * x̂[τ]^2 / k̂[τ] for τ = 2:lookahead+1])

        # variables
        uᶜ = Variable(lookahead)
        uᵏ = Variable(lookahead)
        C = Variable(lookahead + 1)
        k = Variable(lookahead + 1)

        J = 0
        for τ=1:lookahead
            P = diagm(base_diag_cost)
            J += quadform([C[τ]; k[τ]; uᶜ[τ]; uᵏ[τ]], P)
            J -= 2(P[1,1] * C[τ] * Ĉ[τ] + P[2,2] * k[τ] * k̂[τ])
        end

        # constraints
        cons = Vector{Constraint}([
            C[1] == Cₜ,
            k[1] == kₜ,
            C[2:end] == C[1:end-1] + uᶜ,
            k[2:end] == k[1:end-1] + uᵏ,
        ])

        prob = minimize(J, cons)
        solve!(prob, solver(), silent_solver = true)

        return uᶜ.value[1], uᵏ.value[1]
    end
    return mpc_policy
end
