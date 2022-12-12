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

function resample(w::Vector{Float64},N::Int64=length(w))
    return sample(1:length(w),Weights(w),N)
end

#resample(w::Vector{Float64}) = resample(Random.GLOBAL_RNG,w)

## PARTICLE FILTERS ###########################################################

export particle_filter,particle_filter!,bootstrap_filter,bootstrap_filter!

# initialize the particle filter
function particle_filter(
        N::Int64,
        y::Float64,
        model::StateSpaceModel,
        proposal
    )

    # initialize states and reweight
    x    = preallocate(model,N)
    logw = zeros(Float64,N)

    for i in 1:N
        x[i]    = rand(initial_dist(model))
        logw[i] = logpdf(observation(model,x[i]),y)
    
        if !isnothing(proposal)
            logw[i] += logpdf(initial_dist(model),x[i])
          # logw += -1*logpdf.(proposal.(xp),x)
        end
    end

    # normalize initial weights
    logμ,w,_ = normalize(logw)

    return x,w,logμ
end

function particle_filter!(
        states::Vector{XT},
        weights::Vector{Float64},
        y::Float64,
        model::StateSpaceModel,
        proposal
    ) where XT

    ## local vector for log weights
    logw = similar(weights)

    ## resample
    a  = resample(weights)
    x  = states
    xp = deepcopy(x[a])

    ## propagate
    for i in 1:length(states)
        x[i]     = rand(proposal(model,xp[i]))
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
        N::Int64,
        y::Float64,
        model::StateSpaceModel
    )
    # initialize states and reweight
    x    = preallocate(model,N)
    logw = zeros(Float64,N)

    for i in 1:N
        x[i]    = rand(initial_dist(model))
        logw[i] = logpdf(observation(model,x[i]),y)
    end

    # normalize initial weights
    logμ,w,_ = normalize(logw)

    return x,w,logμ
end

function bootstrap_filter!(
        states::Vector{XT},
        weights::Vector{Float64},
        y::Float64,
        model::StateSpaceModel
    ) where XT
    ## local vector for log weights
    logw = similar(weights)

    ## resample
    a  = resample(weights)
    x  = states
    xp = deepcopy(x[a])

    ## propagate
    for i in 1:length(states)
        x[i]     = rand(transition(model,xp[i]))
        logw[i]  = logpdf(observation(model,x[i]),y)
    end

    ## reweight
    return normalize(logw)
end

# there is a better way to write this... no excuses
function log_likelihood(
        N::Int64,
        y::Vector{Float64},
        model::StateSpaceModel
    )
    
    logZ = 0.0
    x,w,logZ = bootstrap_filter(N,y[1],model)

    for t in 2:length(y)
        logμ,w,_ = bootstrap_filter!(x,w,y[t],model)
        logZ    += logμ
    end

    return x,w,logZ
end
