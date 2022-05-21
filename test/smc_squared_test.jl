include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase

# toy model definition A = 0.6 amd B = 0.8
test_model = LinearGaussian(
    Matrix(0.6I(1)),Matrix(0.8I(1)),
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    zeros(1),Matrix(1.0I(1))
)

ssm_test = StateSpaceModel(test_model)
x_test,y_test = simulate(MersenneTwister(1234),ssm_test,100)


# define the constructor for that toy model
model(θ) = StateSpaceModel(LinearGaussian(
    Matrix(θ[1]I(1)),Matrix(θ[2]I(1)),
    Matrix(exp(θ[3])I(1)),Matrix(exp(θ[4])I(1)),
    zeros(1),Matrix(1.0I(1))
))

θ1 = [
    Matrix(0.8I(1)),Matrix(0.5I(1)),
    Matrix(1.0I(1)),Matrix(1.0I(1))
]

θ1 = [0.8,0.5,1.0,1.0]

# finish the prior formatting/structure
prior_func(θ) = product_distribution([
    TruncatedNormal(θ[1],1.0,-1.0,1.0),
    TruncatedNormal(θ[2],1.0,-1.0,1.0),
    Normal(0.0,2.0),
    Normal(0.0,2.0)
])

# ALMOST DONE!!! Run this with the debugger to check the distribution being constructed

smc2 = SMC²(20,100,θ1,prior_func,model,0.1,2)
update_importance!(smc2,y_test[1])