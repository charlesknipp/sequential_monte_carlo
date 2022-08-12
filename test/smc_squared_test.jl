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


alg = SMC²(512,100,prior,mod_func,3,0.8,Random.GLOBAL_RNG)

test_mod = mod_func([0.5,1.0,1.0])
x,y = simulate(test_mod,100)

for i in 1:100
    update!(alg,y)
end

# double check that particle filters indexed properly
for i in 1:100 println(alg.filters[i].state.t[]) end

# find the weighted mean of the particles
sum(
    alg.parameters.x .* alg.parameters.w,
    dims = 1
)
