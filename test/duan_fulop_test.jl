include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase

test_params = LinearGaussian(0.8,0.5,1.0,1.0)
test_model  = StateSpaceModel(test_params)
x,y = simulate(test_model,1000)

prior(μ,Σ) = TruncatedMvNormal(μ,Σ,[-1.0,-1.0,-Inf,-Inf],[1.0,1.0,Inf,Inf])
linGauss(A,B,Q,R) = LinearGaussian(A,B,exp(Q),exp(R))

θ,w = densityTemperedSMC(100,100,y,prior,[0.3,0.4,0.0,0.0],linGauss)

meanθ = mean(reduce(hcat,θ),weights(w),2)
meanθ[3:4] = exp.(meanθ[3:4])

# print output to console
println()
for i in 1:4; @printf("θ[%d] = % 2.6f\n",i,meanθ[i]); end