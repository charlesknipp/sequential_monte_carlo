include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase

test_params = LinearGaussian(0.8,0.5,1.0,1.0)
test_model  = StateSpaceModel(test_params)
x,y = simulate(test_model,50)

prior(μ,Σ) = TruncatedMvNormal(μ,Σ,[-1.0,-1.0,0.0,0.0],[1.0,1.0,Inf,Inf])
θ = densityTemperedSMC(100,1000,y,[0.7,0.7,1.0,1.0],prior)

mean(reduce(hcat,θ.x),weights(θ.w),2)
mean(reduce(hcat,θ.x),dims=2)