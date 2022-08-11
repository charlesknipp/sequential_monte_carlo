include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))


using Distributed
@everywhere using .SequentialMonteCarlo
@everywhere using Printf,Random,Distributions

lg = StateSpaceModel(LinearGaussian(1.0,1.0,1.0,1.0,0.0,1.0),(1,1))
_,y = simulate(lg,10)

pf = ParticleFilter(100,lg,1.0)
logZ = log_likelihood(pf,y)

## PARALLELIZING ##############################################################

rng = Random.GLOBAL_RNG
M = 10
N = 100

prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
])

mod_func(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0,1.0),
    (1,1)
)

θ  = Particles{Vector{Float64}}([rand(rng,prior) for _ in 1:M])
pf = [ParticleFilter(rng,N,mod_func(θ.x[i])) for i in 1:M]

pool = CachingPool(workers())

## BENCHMARKING ###############################################################

using BenchmarkTools

# benchmarks suggest that this is faster than the original implementation taken
# from LowLevelParticleFilters.jl

# single threaded process
@benchmark map(x -> log_likelihood(x,$y),$pf)

# multithreaded process
@benchmark pmap(x -> log_likelihood(x,$y),pool,$pf)
