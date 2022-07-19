include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StaticArrays

test_model = LinearGaussian(
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    zeros(1),Matrix(1.0I(1))
)

ssm_test = StateSpaceModel(test_model)

pf_test = ParticleFilter(100,ssm_test,0.8,MersenneTwister(1234))
x_test,y_test = simulate(MersenneTwister(1234),ssm_test,10)

log_likelihood(pf_test,y_test)

using StatsBase

reset!(pf_test)
for t = eachindex(y_test)
    update!(pf_test,y_test[t])
    xt = reduce(hcat,pf_test.state.x)
    x_mean = mean(xt,weights(pf_test.state.w),2)
    println(@sprintf("simulated: %.5f\tfiltered: %.5f",x_test[t][1],x_mean[1]))
end