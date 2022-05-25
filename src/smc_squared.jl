export SMC²,reset!,random_walk,metropolis,rejuvenate!,update_importance!,reset!

#=
    SMC² is still a work in progress, but it is well on its way to completion.
    The current object keeps track of the particle objects and uses a function
    defined at each iteration to calculate the likelihood for each θ particle.

    log_likelihood_fun() constructs this logZ function and performs it at each
    step to weight each particle. It is not a perfect implementation, but it is
    less intrusive and stores less in memory than before. This is also more
    flexible and easier to debug than before.

    Its implementation is incomplete so there is no working example, but you can
    look at test/smc_squared_test.jl to see how it works for one step forward in
    time.
=#

struct SMC²{ΘT,PΘ,SSM}
    params::ΘT
    prior::PΘ
    model::SSM

    N::Int
    chain_len::Int

    resample_threshold::Float64
    rng::AbstractRNG
end

# default constructor, which establishes the algorithm at t=0
function SMC²(M::Int,N::Int,θ0,prior,model,B,chain_len,rng=Random.GLOBAL_RNG)
    mod_type = eltype(prior(θ0))
    dimθ = length(prior(θ0))

    θprev = Vector{SVector{dimθ,mod_type}}([rand(rng,prior(θ0)) for m=1:M])
    θ = deepcopy(θprev)

    logw = fill(0.0,M)
    w = fill(1.0,M)

    θ  = Particles(θ,θprev,logw,w,Ref(0.),collect(1:M),Ref(1))

    return SMC²(θ,prior(θ0),model,N,chain_len,B,rng)
end

# random walk kernel using the covariance of the particle set
function random_walk(θ0::Particles)
    x  = reduce(hcat,θ0.x)
    Σ  = cov(x,weights(θ0.w),2)
    dθ = (2.38)^2 / length(θ0.x[1])     # optimal scaling parameter

    # returns a function that takes a vector θ
    return θ -> MvNormal(θ,dθ*Σ)
end

# naive random walk which just uses an identity matrix as covariance
function naive_random_walk(θ0)
    dθ = (2.38)^2 / length(θ0.x[1])     # optimal scaling parameter

    # returns a function that takes a vector θ
    return θ -> MvNormal(θ,dθ*I(length(θ0)))

# takes a single particle as it's argument [does not work]
function metropolis(logprob,chain_len,θ0,mcmc_kernel)
    # here logprob is the smc kernel: logZ(θ[m]) + logpdf(p(θ0),θ[m])
    θ  = Vector{typeof(θ0)}(undef,chain_len)
    ll = Vector{Float64}(undef,chain_len)

    θ[1]  = θ0
    ll[1] = logprob(θ0)

    # MH process, relatively easy to follow
    for i = 2:chain_len
        θi = rand(mcmc_kernel(θ[i-1]))
        lli = logprob(θi)
        if rand() < exp(lli-ll[i-1])
            θ[i] = θi
            ll[i] = lli
        else
            θ[i] = θ[i-1]
            ll[i] = ll[i-1]
        end
    end

    return θ[chain_len]
end

# resample parameter particles
function resample!(smc²::SMC²)
    θ = smc².params
    a = wsample(smc².rng,1:length(θ),θ.w,length(θ))

    for i = eachindex(θ.x)
        θ.x[i] = θ.xprev[a[i]]
    end

    return θ.x
end

# rejuvenation by way of MH steps [does not work]
function rejuvenate!(smc²::SMC²,logprob)
    θ = smc².params
    #mcmc_kernel = random_walk(smc².params)
    mcmc_kernel = naive_random_walk(θ.x[1])

    for i = eachindex(θ.x)
        θ.x[i] = metropolis(logprob,smc².chain_len,θ.xprev[i],mcmc_kernel)
    end

    return θ.x
end

function reset_weights!(smc²::SMC²)
    fill!(smc².params.logw,0.0)
    fill!(smc².params.w,1.0)
    smc².params.maxw[] = 0.0
end

@inline normalize!(smc²::SMC²) = normalize!(smc².params)

# this is defines a single step t of the algorithm
function update_importance!(smc²::SMC²,y)
    # define a bootstrap filter with the new set of particles
    function bootstrap_filter(θ,pf=nothing)
        mod = smc².model(θ)
        return ParticleFilter(smc².N,mod)
    end

    # define the likelihood calculation by the above bootstrap filter
    θ = smc².params
    logZ = log_likelihood_fun(bootstrap_filter,smc².prior,y[1:θ.t[]])

    # reweight by adding the log likelihood to each log weight
    for i in eachindex(θ.x)
        smc².params.logw[i] += logZ(smc².params.x[i])
    end

    # calculate the ESS for each time t
    normalize!(smc²)
    ess = sum(smc².params.w)^2 / sum(abs2,smc².params.w)

    # resample step
    if ess < smc².resample_threshold*length(θ)
        println(ess,"\t[rejuvenating]")

        resample!(smc²)
        rejuvenate!(smc²,logZ)

        reset_weights!(smc²)
    else
        println(ess)
    end

    copyto!(smc².params.xprev,smc².params.x)
    θ.t[] += 1
end

# reset after running it T times
function reset!(smc²::SMC²)
    θ = smc².params

    for i = eachindex(θ.xprev)
        θ.xprev[i] = rand(smc².rng,smc².prior)
        θ.x[i] = copy(θ.xprev[i])
    end

    fill!(θ.logw,0.0)
    fill!(θ.w,1.0)

    smc².params.t[] = 1
end