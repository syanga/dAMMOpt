
""" MPC policy """
function make_mpc_policy(base_diag_cost, lookahead, λ_k; use_offdiagonal_costs = true, solver = ECOS.Optimizer)
    @assert lookahead > 0
    @assert length(base_diag_cost) == 4
    α, β, γ, δ = base_diag_cost
    function mpc_policy(res, forecast, est_net_position)
        Cₜ, kₜ, p̂ₒ, x̂ = res.C[end], res.k[end], forecast[:, 4], forecast[:, 2]

        # nominal trajectories k̂, Ĉ
        k̂ = [kₜ + λ_k * τ for τ = 0:lookahead]
        Ĉ = vcat(Cₜ, [p̂ₒ[τ] * x̂[τ]^2 / k̂[τ] for τ = 2:lookahead+1])

        # variables
        C = Variable(lookahead + 1)
        k = Variable(lookahead + 1)
        uᶜ = Variable(lookahead)
        uᵏ = Variable(lookahead)

        # preconditioning
        Ts = []
        cost_scaling = lookahead

        J = 0
        for τ = 1:lookahead
            if use_offdiagonal_costs
                a = -est_net_position[τ] / (24 * x̂[τ]^2)
                b = -1 / (x̂[τ] + est_net_position[τ])
                c = est_net_position[τ] / (x̂[τ] * (x̂[τ] + est_net_position[τ]))
            else
                a = 0
                b = 0
                c = 0
            end

            P = [
                2α a 0 b
                a 2β c 0
                0 c 2γ b
                b 0 b 2δ
            ] / 2

            # make matrix PSD
            λₘ = minimum(eigvals(P))
            if λₘ < 0
                P = P + Diagonal(max(1.1abs(λₘ), 1e-9) * ones(4))
            end

            # linear cost term
            lin = [-2(P[1, 1] * Ĉ[τ]); -2(P[2, 2] * k̂[τ]); 0; 0]

            # preconditioning matrix if applicable
            T = I(4)

            z = [C[τ]; k[τ]; uᶜ[τ]; uᵏ[τ]]
            J += quadform(z, Symmetric(T' * P * T)) + dot(T' * lin, z)
            push!(Ts, T)
        end

        # constraints
        cons = Vector{Constraint}([Ts[1][1:2, :] * [C[1]; k[1]; uᶜ[1]; uᵏ[1]] == [Cₜ; kₜ]])
        for τ = 1:lookahead
            z⁺ = [C[τ+1]; k[τ+1]; 0; 0]
            z⁻ = [C[τ]; k[τ]; uᶜ[τ]; uᵏ[τ]]
            push!(cons, Ts[τ][1:2, :] * z⁺ == [1 0 1 0; 0 1 0 1] * Ts[τ] * z⁻)
        end

        prob = minimize(J / cost_scaling, cons)
        solve!(prob, solver(), silent_solver = true)

        # only return control if solver did not fail
        if string(prob.status) in ["OPTIMAL", "ALMOST_OPTIMAL", "SLOW_PROGRESS"]
            z₁ = [C.value[1]; k.value[1]; uᶜ.value[1]; uᵏ.value[1]]
            C_control = dot(Ts[1][3, :], z₁)
            k_control = dot(Ts[1][4, :], z₁)
            return C_control, k_control
        end
        return 0, 0
    end
    return mpc_policy
end
