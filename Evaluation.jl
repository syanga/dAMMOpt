using Plots


""" visually check policy performance """
function visual_eval(f::PerpSimulator, policies, po, x0)
    for idx = 1:length(policies)
        name, policy = policies[idx]
        res = simulate(f, po, x0; policy = policy, forecast_length = forecast_length)

        # check forecast quality
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
    
        plot(res.mark, label = "mark")
        plot!(res.oracle, label = "oracle")
        xlabel!("Time")
        ylabel!("Price")
        savefig("assets/mark_vs_oracle_$(name).png")

        plot(res.net_position, label = "net position (long - short)")
        plot!(res.divergence * 10, label = "scaled and signed price divergence")
        savefig("assets/position_vs_divergence_$(name).png")

        plot(res.revenue, label = "revenue")
        xlabel!("Time")
        ylabel!("Period revenue")
        savefig("assets/funding_revenue_$(name).png")

        net_rev = sum(res.revenue)
        mse_div = mean((res.mark .- res.oracle) .^ 2)
        mean_slippage = mean(res.unit_slippage)
        println(
            "net revenue = $(net_rev), funding rev = $(-sum(res.funding)), mse div = $(mse_div), mean slippage = $(mean_slippage)",
        )
    end
end



""" Evaluate average performance of different policies """
function evaluate_policies(f::PerpSimulator, policies, x0; num_oracle_trajectories = 10)
    # compute metrics given simulation result
    function _metrics(res)
        rev = sum(res.revenue)
        mean_abs_div = mean(abs.((res.mark .- res.oracle) ./ res.oracle))
        mean_slippage = mean(res.unit_slippage)
        return [rev, mean_abs_div, mean_slippage]
    end

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
