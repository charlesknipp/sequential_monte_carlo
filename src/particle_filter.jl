struct ParticleFilter{ST} <: AbstractFilter
    model::StateSpaceModel
    state::ST

    resample_threshold::Float64
    rng::AbstractRNG
end

function ParticleFilter(N::Integer,model,B=0.1,rng=Random.GLOBAL_RNG)
    mod_type = eltype(model.initial_dist)
    dim_x = length(model.initial_dist)

    xprev = Vector{SVector{dim_x,mod_type}}([rand(rng,model.initial_dist) for n=1:N])
    x = deepcopy(xprev)

    logw = fill(-1*log(N),N)
    w = fill(1/N,N)

    s = Particles(x,xprev,logw,w,Ref(0.),collect(1:N),Ref(1))

    return ParticleFilter(model,s,B,rng)
end


Base.@propagate_inbounds function reweight!(pf::ParticleFilter,y)
    logw = pf.state.logw
    dist = pf.model.observation
    any(ismissing,y) && return logw

    if dist isa UnivariateDistribution && length(y) == 1
        for i = 1:length(pf.state)
            # not sure if logw += logpdf(⋅) or logw = logpdf(⋅)
            logw[i] = logpdf(dist(pf.state.x[i][1]),y[1])
        end
    else
        for i = 1:length(pf.state)
            # not sure if logw += logpdf(⋅) or logw = logpdf(⋅)
            logw[i] = logpdf(dist(pf.state.x[i]),y)
        end
    end

    return logw
end

# normalizes weights and updates particle cloud (too complicated)
function logsumexp!(logw,w,maxw=Ref(zero(eltype(logw))))::eltype(logw)
    offset,maxind = findmax(logw)
    logw .-= offset

    # normalize new weights
    LoopVectorization.vmap!(exp,w,logw)
    sumw   = sum_all_but(w,maxind)
    w    .*= 1/(sumw+1)
    logw .-= log1p(sumw)

    # adjusted maximum log weight
    maxw[] += offset

    return log1p(sumw) + maxw[] - log(length(logw))
end

@inline logsumexp!(p) = logsumexp!(p.logw,p.w,p.maxw)
@inline logsumexp!(pf::AbstractFilter) = logsumexp!(pf.state)


# resampling required
Base.@propagate_inbounds function propagate!(pf::ParticleFilter,a::Vector{Int})
    s = pf.state
    transition = pf.model.transition
    x,xp = s.x,s.xprev

    vec_type = eltype(x)
    d_dims   = length(vec_type)
    xprop    = zeros(d_dims)

    for i = eachindex(x)
        x[i] = vec_type(rand!(pf.rng,transition(xp[a[i]]),xprop))
    end

    return x
end

# no need for resampling
Base.@propagate_inbounds function propagate!(pf::ParticleFilter)
    s = pf.state
    transition = pf.model.transition
    x,xp = s.x,s.xprev

    vec_type = eltype(x)
    d_dims   = length(vec_type)
    xprop    = zeros(d_dims)

    for i = eachindex(x)
        x[i] = vec_type(rand!(pf.rng,transition(xp[i]),xprop))
    end

    return x
end

index(pf::AbstractFilter) = pf.state.t[]


function correct!(pf,y)
    # calculates log weights
    reweight!(pf,y)

    # normalizes weights and finds the likelihood
    ll = logsumexp!(pf.state)

    return ll
end

function predict!(pf)
    particles = pf.state
    N = length(particles)

    # rethink this in terms of particle types/filter types
    if ESS(particles) < pf.resample_threshold*N
        a = wsample(pf.rng,1:N,particles.w,N)
        propagate!(pf,a)
        reset_weights!(particles)
    else # Resample not needed
        particles.a .= 1:N
        propagate!(pf)
    end

    # move forward in time
    copyto!(particles.xprev,particles.x)
    pf.state.t[] += 1
end

function update!(pf::AbstractFilter,y)
    ll = correct!(pf,y)
    predict!(pf)

    return ll
end

function reset!(pf::AbstractFilter)
    particles = pf.state

    for i = eachindex(particles.xprev)
        particles.xprev[i] = rand(pf.rng,pf.model.initial_dist)
        particles.x[i] = copy(particles.xprev[i])
    end

    fill!(particles.logw,-log(length(particles)))
    fill!(particles.w,1/length(particles))

    pf.state.t[] = 1
end