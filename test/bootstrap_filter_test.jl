include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))
using .SequentialMonteCarlo
using Printf

test_params = LinearGaussian(0.9,1.0,1.0,1.0)
test_model  = StateSpaceModel(test_params)

x,y  = simulate(test_model,100)
xs = bootstrapFilter(1000,y,test_model)

for t in 1:100
    μ = sum(xs.p[t].x)/1000
    println(@sprintf("x_pf: %.5f\tx_sim: %.5f",μ,x[t]))
end
