include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase


prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
])

mod_func(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,exp(θ[2]),exp(θ[3]),0.0,1.0),
    (1,1)
)


alg = SMC²(100,100,prior,mod_func,4,0.5,Random.GLOBAL_RNG)

test_mod = mod_func([0.5,1.0,1.0])
x,y = simulate(test_mod,100)

for i in 1:100
    update!(alg,y)
end
