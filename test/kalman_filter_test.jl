include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase

# toy model definition A = 0.6 amd B = 0.8
test_model = LinearGaussian(
    Matrix(0.6I(1)),Matrix(0.8I(1)),
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    zeros(1),Matrix(1.0I(1))
)

T = 100
ssm_test = StateSpaceModel(test_model)
x_test,y_test = simulate(MersenneTwister(1234),ssm_test,T)


# define the constructor for that toy model
model(θ) = StateSpaceModel(LinearGaussian(
    Matrix(θ[1]I(1)),Matrix(θ[2]I(1)),
    Matrix(exp(θ[3])I(1)),Matrix(exp(θ[4])I(1)),
    zeros(1),Matrix(1.0I(1))
))

# write kalman filter routine here...