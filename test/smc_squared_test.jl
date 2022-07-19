include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase

# toy model definition A = 0.6 amd B = 0.8
test_model = LinearGaussian(
    Matrix(0.6I(1)),Matrix(0.8I(1)),
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    zeros(1),Matrix(1.0I(1))
)

T = 20
ssm_test = StateSpaceModel(test_model)
x_test,y_test = simulate(MersenneTwister(1234),ssm_test,T)


# define the constructor for that toy model
model(θ) = StateSpaceModel(LinearGaussian(
    Matrix(θ[1]I(1)),Matrix(θ[2]I(1)),
    Matrix(exp(θ[3])I(1)),Matrix(exp(θ[4])I(1)),
    zeros(1),Matrix(1.0I(1))
))


## NOTE: eventually accept a list of lists for θ, right now it does not work,
##       but it should be relatively easy to implement once the alg is tested
##       and working
##
## θ1 = [
##     Matrix(0.8I(1)),Matrix(0.5I(1)),
##     Matrix(1.0I(1)),Matrix(1.0I(1))
## ]


#θ1 = [0.8,0.5,1.0,1.0]

# prior works relatively well
prior = product_distribution([
    TruncatedNormal(0.0,1.0,-1.0,1.0),
    TruncatedNormal(0.0,1.0,-1.0,1.0),
    Normal(0.0,2.0),
    Normal(0.0,2.0)
])

# initialize the algorithm at t=0
smc2 = SMC²(100,100,prior,model,0.8,3)

# to run the algorithm perform the following:
for t in 1:T
    #θ_t  = smc2.params
    #pf_t = smc2.filters

    #show(θ_t.x)

    # [smc2.filters[i].state.t[] for i in 1:100]
    update_importance!(smc2,y_test)
end

# test a regular particle filter here
θ_test = [0.6,0.8,1.0,1.0]
pf_test = ParticleFilter(100,model(θ_test),Inf,MersenneTwister(1234))
log_likelihood(pf_test,y_test)

#=
smc2.params.x
smc2.likelihood
update_importance!(smc2,y_test)
=#

predθ = vec(mean(reduce(hcat,smc2.params.x),weights(smc2.params.w),2))
predθ[3:4] = exp.(predθ[3:4])

using Plots

# histogram(smc2.params.x)

# reset after the algorithm is finished running
reset!(smc2)

predθ