export bootstrap_filter,bootstrap_filter!,log_likelihood

function reweight(logw::Vector{Float64})
    maxw = maximum(logw)
    w    = exp.(logw.-maxw)
    sumw = sum(w)

    logμ = maxw + log(sumw) - log(length(logw))
    w    = w/sumw
    ess  = 1.0/sum(w.^2)

    return (logμ,w,ess)
end

function resample(rng::AbstractRNG,w::Vector{Float64},N::Int64=length(w))
    return sample(rng,1:length(w),Weights(w),N)
end

resample(w::Vector{Float64},N::Int64=length(w)) = resample(Random.GLOBAL_RNG,w,N)

## PARTICLE FILTERS ###########################################################

# initialize the particle filter
function bootstrap_filter(
        rng::AbstractRNG,
        N::Int64,
        y::Float64,
        model::StateSpaceModel
    )

    ## initialize states
    x    = preallocate(model,N)
    logw = zeros(Float64,N)

    ## propagate
    for i in 1:N
        x[i]    = rand(rng,initial_dist(model))
        logw[i] = logpdf(observation(model,x[i]),y)
    end

    ## reweight
    maxw = maximum(logw)
    w    = exp.(logw.-maxw)
    sumw = sum(w)

    logμ = maxw + log(sumw) - log(length(logw))
    w    = w/sumw

    return x,w,logμ
end

function bootstrap_filter(
        N::Int64,
        y::Float64,
        model::StateSpaceModel
    )
    bootstrap_filter(Random.GLOBAL_RNG,N,y,model)
end

function bootstrap_filter!(
        rng::AbstractRNG,
        states::Vector{XT},
        weights::Vector{Float64},
        y::Float64,
        model::StateSpaceModel
    ) where XT

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
    end

    ## reweight
    maxw = maximum(logw)
    w    = exp.(logw.-maxw)
    sumw = sum(w)

    logμ = maxw + log(sumw) - log(length(logw))
    w    = w/sumw

    return logμ,w
end

function bootstrap_filter!(
        states::Vector{XT},
        weights::Vector{Float64},
        y::Float64,
        model::StateSpaceModel
    ) where XT
    bootstrap_filter!(Random.GLOBAL_RNG,states,weights,y,model)
end

function log_likelihood(
        rng::AbstractRNG,
        N::Int64,
        y::Vector{Float64},
        model::StateSpaceModel
    )

    logZ = 0.0
    x,w,logZ = bootstrap_filter(rng,N,y[1],model)

    for t in 2:length(y)
        logμ,w = bootstrap_filter!(rng,x,w,y[t],model)
        logZ  += logμ
    end

    return x,w,logZ
end

log_likelihood(N::Int64,y::Vector{Float64},model::StateSpaceModel) = log_likelihood(Random.GLOBAL_RNG,N,y,model)
