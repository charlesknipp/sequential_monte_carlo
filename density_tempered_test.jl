using LinearAlgebra,Statistics,Random,Distributions
using ProgressMeter

cd("src")

# import helper functions
include("dynamic_model.jl")
include("particle_filter.jl")
include("truncated_mv_normal.jl")
include("density_tempered_smc.jl")

########################## TESTING BLOCK 1 ##########################

# it is good convention to set a seed for testing
Random.seed!(1234)

# recall: set A = .8 instead of an edge case
sims = NDLM(0.8,1.0,1.0,1.0)
_,y = simulate(200,sims)

# guess on the true parameters to see if this works...
guess = [0.5,0.5,.81,.81]
θ,ξ = densityTemperedSMC(200,65,100,y,guess)
map(i -> mean(θ[length(θ)][i,:]),1:4)

#####################################################################