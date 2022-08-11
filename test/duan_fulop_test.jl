include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase

prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
])

mod_func(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0,1.0),
    (1,1)
)

test_mod = mod_func([0.5,1.0,1.0])
x,y = simulate(test_mod,500)

# this algorithm seems to function as intended
θ = density_tempered_pf(512,1024,y,prior,mod_func,10,0.5,Random.GLOBAL_RNG)
mean(reduce(hcat,θ.x)',dims=1)
