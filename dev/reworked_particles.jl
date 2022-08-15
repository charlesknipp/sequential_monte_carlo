using StatsBase,Distributions,Random,LinearAlgebra
include("state_space_models.jl")

## PARTICLES ##################################################################

struct Particles{T<:Union{Float64,Vector{Float64}}}
    x::Vector{T}
    w::Vector{Float64}

    logμ::Base.RefValue{Float64}
    ess::Base.RefValue{Float64}

    function Particles{T}(x::Vector{T}) where T <: Union{Float64,Vector{Float64}}
        N = length(x)

        return new(
            x,
            fill(1/N,N),
            Ref(-log(N)),
            Ref(1.0)
        )
    end
end

Base.length(X::Particles) = length(X.w)

function reweight!(X::Particles,logw::Vector{Float64})
    maxw = maximum(logw)
    w    = exp.(logw.-maxw)
    sumw = sum(w)

    # change the "immutable" struct by RefValue manipulation
    X.logμ[] = maxw + log(sumw) - log(length(logw))
    X.w[:]  .= w/sumw
    X.ess[]  = 1.0/sum(X.w.^2)

    return X.logμ[]
end

function resample!(rng::AbstractRNG,X::Particles)
    N = length(X)
    return sample(rng,1:N,Weights(X.w),N)
end

resample!(X::Particles) = resample!(Random.GLOBAL_RNG,X)
particle_type(X::Particles{T}) where {T} = T

## PARTICLE FILTERS ###########################################################

#=
NOTES: I need to perform a particle filter step at time t, lets call this func-
    tion `f(θ)`. We need a few things of function f...

    - f takes either a model or a vector of parameters, an observation y[t],
      a proposal (for auxiliary filters and vsmc), and particles x
    - f returns a particles object with logμ, weights, and states
=#

function particle_filter!(
        rng::AbstractRNG,
        states::Particles,
        y::Float64,
        model::StateSpaceModel,
        proposal
    )

    ## local vector for log weights
    logw = similar(states.w)

    ## resample
    a  = resample!(rng,states)
    x  = states.x
    xp = deepcopy(x[a])

    ## propagate
    for i in 1:length(states)
        x[i]     = rand(rng,transition(model,xp[i]))
        logw[i]  = logpdf(observation(model,x[i]),y)

        if !isnothing(proposal)
            logw[i] += logpdf(transition(model,xp[i]),x[i])
            logw[i] += -1*logpdf(proposal(xp[i]),x[i])
        end
    end

    ## reweight
    return reweight!(states,logw)
end


function bootstrap_filter!(
        rng::AbstractRNG,
        states::Particles,
        y::Float64,
        model::StateSpaceModel
    )
    ## set proposal to nothing
    return particle_filter(rng,states,y,model,nothing)
end

particle_filter!(states::Particles,y::Float64,model::StateSpaceModel,proposal) = particle_filter(Random.GLOBAL_RNG,states,y,model,proposal)
bootstrap_filter!(states::Particles,y::Float64,model::StateSpaceModel) = bootstrap_filter(Random.GLOBAL_RNG,states,y,model)

# there is a better way to write this... no excuses
function log_likelihood(
        rng::AbstractRNG,
        states::Particles,
        y::Vector{Float64},
        model::StateSpaceModel,
        proposal
    )
    logZ = 0.0

    for t in eachindex(y)
        logZ += particle_filter!(rng,states,y[t],model,proposal)
    end

    return logZ
end

## TESTING ####################################################################

lg = StateSpaceModel(LinearGaussian(1.0,1.0,1.0,1.0,0.0,1.0),(1,1))
_,y = simulate(lg,10)

states = Particles{Float64}([rand(initial_dist(lg)) for n=1:100])

logZ = bootstrap_filter(Random.GLOBAL_RNG,states,y[1],lg)
logZ += bootstrap_filter(Random.GLOBAL_RNG,states,y[2],lg)
logZ += bootstrap_filter(Random.GLOBAL_RNG,states,y[3],lg)
logZ += bootstrap_filter(Random.GLOBAL_RNG,states,y[4],lg)
logZ += bootstrap_filter(Random.GLOBAL_RNG,states,y[5],lg)
logZ += bootstrap_filter(Random.GLOBAL_RNG,states,y[6],lg)
