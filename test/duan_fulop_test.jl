include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase

test_params = LinearGaussian(0.8,0.5,1.0,1.0)
test_model  = StateSpaceModel(test_params)
x,y = simulate(test_model,50)

prior(μ,Σ) = TruncatedMvNormal(μ,Σ,[-1.0,-1.0,0.0,0.0],[1.0,1.0,Inf,Inf])
linGauss(A,B,Q,R) = LinearGaussian(A,B,Q,R)

θ,w = densityTemperedSMC(100,100,y,prior,[0.3,-0.4,1.0,1.0],linGauss)

mean(reduce(hcat,θ),dims=2)