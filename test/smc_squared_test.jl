include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase


# simulate data
test_params = LinearGaussian(0.8,0.5,1.0,1.0)
test_model  = StateSpaceModel(test_params)
x,y = simulate(test_model,50)

# construct a prior distribution
prior(μ,Σ) = TruncatedMvNormal(μ,Σ,[-1.0,-1.0,0.0,0.0],[1.0,1.0,Inf,Inf])
θ0 = [0.7,0.7,1.0,1.0]

θ,Xt = SMC2(20,100,y,θ0,prior,0.5,LinearGaussian)

mean(reduce(hcat,θ.x),weights(θ.w),2)

# [println(Xtm.logμ) for Xtm in Xt.p]

#=
    The output looks as it should, but the weights that are generated are not
    what they seem. The algorithm spits out an evenly weighted particle cloud
    which should not be the case
=#