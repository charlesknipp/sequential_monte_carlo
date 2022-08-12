export ParticleFilter,update!,reset!,log_likelihood

#=
ParticleFilter, subclass of AbstractFilter, is relatively self explainatory. It
defines a class of particle filters using a state space model and a particle
set. Currently updates to the filter must be defined for different dimensions
because initializing and propogating a particle filter varies depending on the
type of the state attribute.

For particle filtering, we define the following methods:
    update!(pf::ParticleFilter,y::particle_type(pf.state))
    reset!(pf::ParticleFilter)
    log_likelihood(pf::ParticleFilter,y::Vector{particle_type(pf.state)})
=#

abstract type AbstractFilter end

struct ParticleFilter <: AbstractFilter
    model::StateSpaceModel
    state::Particles

    resample_threshold::Float64
    rng::AbstractRNG
end

# default constructor given number of particles N, resample threshold B, and rng
function ParticleFilter(rng::AbstractRNG,N::Int,model,B=1.0)
    modtype = model.dims[1] == 1 ? Float64 : Vector{Float64}
    state = Particles{modtype}([rand(rng,initial_dist(model)) for n=1:N])

    return ParticleFilter(model,state,B,rng)
end

ParticleFilter(N::Int,model,B=1.0) = ParticleFilter(Random.GLOBAL_RNG,N,model,B)

# create two versions for y === Float64 and y === Vector{Float64}
function update!(pf::ParticleFilter,y::Float64)
    xp   = copy(pf.state.x[pf.state.a])
    logw = similar(pf.state.w)

    ## propogate
    for i in eachindex(xp)
        pf.state.x[i] = rand(pf.rng,transition(pf.model,xp[i]))
        logw[i] = logpdf(observation(pf.model,pf.state.x[i]),y)
    end

    ## resample
    log_likelihood = reweight!(pf.state,logw)
    resample!(pf.state)

    pf.state.t[] += 1

    return log_likelihood
end

function reset!(pf::ParticleFilter)
    # resample from the initial distribution
    for i in eachindex(pf.state.x)
        pf.state.x[i] = rand(pf.rng,initial_dist(pf.model))
    end

    # reset the weights
    fill!(pf.state.w,1/length(pf.state))
    
    # reset particle characteristics
    pf.state.ess[]  = 0.0
    pf.state.logμ[] = -log(length(pf.state))
    pf.state.t[]    = 1
end

# again this is for univariate filters...
function log_likelihood(pf::AbstractFilter,y::Vector{Float64})
    reset!(pf)
    return sum(x -> update!(pf,x),y)
end

(pf::ParticleFilter)(y::Float64) = update!(pf,y)
