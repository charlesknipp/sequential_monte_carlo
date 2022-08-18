include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra

test_prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
])

mod_func(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0),
    (1,1)
)

test_θ = [0.5,1.0,1.0]
x,y = simulate(mod_func(test_θ),100)

## density tempered
dt_test = SMC(Random.GLOBAL_RNG,512,1024,mod_func,test_prior,3,0.5)
density_tempered(dt_test,y)
expected_parameters(dt_test)


## smc²
smc²_test = SMC(Random.GLOBAL_RNG,512,1024,mod_func,test_prior,3,0.5)
smc²(smc²_test,y)

for t in 2:100
    smc²!(smc²_test,y,t)
end

expected_parameters(smc²_test)