export ParticleFilter,reweight!,log_likelihood,propagate!,update!,resample!,reset!

# create an abstract type for extension to particle filters and Kalman filters
abstract type AbstractFilter end

#=
    ParticleFilter is another parametric object which keeps track of the move-
    ment along the fitler at each state. The design is rather simple and relies
    heavily on the state element.

    To run this we can use the log_likelihood function to get the log normal-
    izing constant given a particle filter object.

    See test/bootstrap_filter_test.jl for an example of how to use this object.
=#

# create a particle fitler object with a model, state, and rng
struct ParticleFilter <: AbstractFilter
    model::StateSpaceModel
    state::Particles

    resample_threshold::Float64
    rng::AbstractRNG
end

# default constructor given number of particles N, resample threshold B, and rng
function ParticleFilter(N::Int,model,B=0.1,rng=Random.GLOBAL_RNG)
    mod_type = eltype(model.initial_dist)
    dim_x = length(model.initial_dist)

    # preallocate arrays
    xprev = Vector{SVector{dim_x,mod_type}}([rand(rng,model.initial_dist) for n=1:N])
    x = deepcopy(xprev)

    logw = fill(-1*log(N),N)
    w = fill(1/N,N)

    # establish particle set
    s = Particles(x,xprev,logw,w,Ref(0.),collect(1:N),Ref(1))

    return ParticleFilter(model,s,B,rng)
end

# this exclusively calculates the log weights given an observation y[t]
Base.@propagate_inbounds function reweight!(pf::ParticleFilter,yt)
    logw = pf.state.logw
    dist = pf.model.observation
    any(ismissing,yt) && return logw

    # weight the particles as describes in the footnote of SMC² on page 4
    if dist isa UnivariateDistribution && length(yt) == 1
        # for univariate distributions, yt[1] == unnest(yt)
        for i = 1:length(pf.state)
            logw[i] = logpdf(dist(pf.state.x[i][1]),yt[1])
        end
    else
        for i = 1:length(pf.state)
            logw[i] = logpdf(dist(pf.state.x[i]),yt)
        end
    end

    return logw
end


# when x is resampled, propagate forward...
Base.@propagate_inbounds function propagate!(pf::ParticleFilter,a::Vector{Int})
    s = pf.state
    transition = pf.model.transition
    x,xp = s.x,s.xprev

    # define allocations so we can operate on x in-place
    vec_type = eltype(x)
    d_dims   = length(vec_type)
    xprop    = zeros(d_dims)

    # propagate x forward and populate the new set of particles x with index a
    for i = eachindex(x)
        x[i] = vec_type(rand!(pf.rng,transition(xp[a[i]]),xprop))
    end

    return x
end

# if there is no resampling, propagate using this function...
Base.@propagate_inbounds function propagate!(pf::ParticleFilter)
    s = pf.state
    transition = pf.model.transition
    x,xp = s.x,s.xprev

    # define allocations so we can operate on x in-place
    vec_type = eltype(x)
    d_dims   = length(vec_type)
    xprop    = zeros(d_dims)

    # propagate x forward
    for i = eachindex(x)
        x[i] = vec_type(rand!(pf.rng,transition(xp[i]),xprop))
    end

    return x
end

index(pf::AbstractFilter) = pf.state.t[]

# resample and propogate forward
function resample!(pf)
    particles = pf.state
    N = length(particles)

    # resetting the weights here is necessary for the reweight function to work
    if ESS(particles) < pf.resample_threshold*N
        a = wsample(pf.rng,1:N,particles.w,N)
        propagate!(pf,a)
        reset_weights!(particles)
    else
        # recycle all particles in this instance
        particles.a .= 1:N
        propagate!(pf)
    end

    # move forward in time
    copyto!(particles.xprev,particles.x)
    pf.state.t[] += 1
end

function update!(pf::AbstractFilter,yt)
    # reweight particles and calculate likelihood
    reweight!(pf,yt)
    ll = normalize!(pf.state)

    # resample and propogate forward
    resample!(pf)

    return ll
end

# pretty self explainatory reset function
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

function log_likelihood(pf::AbstractFilter,y)
    reset!(pf)
    # should encase y in a tuple; see https://github.com/baggepinnen/LowLevelParticleFilters.jl/blob/master/src/smoothing.jl#L86
    return sum(x -> update!(pf,x),y)
end