# dAMMOpt

Model predictive control for dynamic Automated Market Maker (dAMM) operation. We adjust the parameters of a constant product market maker over time by solving a sequential decision making problem, trading off oracle price tracking, market operation costs, and slippage. See the [https://github.com/syanga/dAMMOpt/dAMMOpt.pdf](https://github.com/syanga/dAMMOpt/blob/main/dAMMOpt.pdf).

The experiments in the report can be run by executing:
```julia
julia dAMMOpt.jl
```
