## PARTICLES ##################################################################

export normalize,resample,log_likelihood

function normalize(logw::Vector{Float64})
    maxw = maximum(logw)
    w    = exp.(logw.-maxw)
    sumw = sum(w)

    logμ = maxw + log(sumw) - log(length(logw))
    w    = w/sumw
    ess  = 1.0/sum(w.^2)

    return (logμ,w,ess)
end

function resample(rng::AbstractRNG,w::Vector{Float64})
    N = length(w)
    return sample(rng,1:N,Weights(w),N)
end

resample(w::Vector{Float64}) = resample(Random.GLOBAL_RNG,w)

## PARTICLE FILTERS ###########################################################

export particle_filter,particle_filter!,bootstrap_filter,bootstrap_filter!

# initialize the particle filter
function particle_filter(
        rng::AbstractRNG,
        N::Int64,
        y::Float64,
        model::StateSpaceModel,
        proposal
    )

    # initialize states and reweight
    x    = rand(rng,initial_dist(model),N)
    logw = logpdf.(observation.(Ref(model),x),y)

    if !isnothing(proposal)
        logw += logpdf.(initial_dist(model),x)
        logw += -1*logpdf.(proposal.(xp),x)
    end

    # normalize initial weights
    logμ,w,_ = normalize(logw)

    return x,w,logμ
end

function particle_filter!(
        rng::AbstractRNG,
        states::Vector{Float64},
        weights::Vector{Float64},
        y::Float64,
        model::StateSpaceModel,
        proposal
    )

    ## local vector for log weights
    logw = similar(weights)

    ## resample
    a  = resample(rng,weights)
    x  = states
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
    return normalize(logw)
end

# initialize the bootstrap filter
function bootstrap_filter(
        rng::AbstractRNG,
        N::Int64,
        y::Float64,
        model::StateSpaceModel
    )
    ## set proposal to nothing
    return particle_filter(rng,N,y,model,nothing)
end

function bootstrap_filter!(
        rng::AbstractRNG,
        states::Vector{Float64},
        weights::Vector{Float64},
        y::Float64,
        model::StateSpaceModel
    )
    ## set proposal to nothing
    return particle_filter!(rng,states,weights,y,model,nothing)
end

# there is a better way to write this... no excuses
function log_likelihood(
        rng::AbstractRNG,
        N::Int64,
        y::Vector{Float64},
        model::StateSpaceModel,
        proposal=nothing
    )
    logZ = 0.0

    x,w,logZ = particle_filter(rng,N,y[1],model,proposal)

    for t in 2:length(y)
        logμ,w,_ = particle_filter!(rng,x,w,y[t],model,proposal)
        logZ    += logμ
    end

    return x,w,logZ
end