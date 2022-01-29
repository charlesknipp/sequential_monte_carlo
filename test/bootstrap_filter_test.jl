include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions

test_params = LinearGaussian(1.0,1.0,1.0,1.0)
test_model  = StateSpaceModel(test_params)

# initializing needs some work
# Random.seed!(123)
x,y = simulate(test_model,100)

# Random.seed!(123)
xs_bf = bootstrapFilter(1000,y,test_model,Inf)
xs_kf = kalmanFilter(y,test_params)

for t in 1:100
    μ = quantile(xs_bf.p[t].x,.50)
    kf = xs_kf[t,2]
    println(@sprintf("x_bf: %.5f\tx_kf: %.5f\tx_sim: %.5f",μ,kf,x[t]))
end
