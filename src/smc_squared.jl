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

struct SMC²{ΘT,PΘ,SSM,K}
    params::ΘT
    prior::PΘ
    model::SSM
    mcmc_kernel::K

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

    logw = fill(-log(M),M)
    w = fill(1/M,M)

    θ  = Particles(θ,θprev,logw,w,Ref(0.),collect(1:M),Ref(1))
    pθ = random_walk(θ)

    return SMC²(θ,prior(θ0),model,pθ,N,chain_len,B,rng)
end

# random walk kernel which makes the likelihood ratio an easier computation
function random_walk(θ0::Particles)
    x = reduce(hcat,θ0.x)

    μ = vec(mean(x,weights(θ0.w),2))
    Σ = cov(x,weights(θ0.w),2)

    # returns a function that takes a vector θ
    return rand(MvNormal(μ,0.1*Σ))
end

# expand to work for θ particles
function metropolis(logprob,chain_len,θ0,mcmc_kernel)
    # here logprob is the smc kernel: logZ(θ[m]) + logpdf(p(θ0),θ[m])
    θ  = Vector{typeof(θ0.x)}(undef,chain_len)
    ll = Vector{Float64}(undef,chain_len)

    θ[1]  = θ0.x
    ll[1] = logprob(θ0)

    # MH process, relatively easy to follow
    for i = 2:chain_len
        θi = mcmc_kernel(θ[i-1])
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

# rejuvenation by way of MH steps
function rejuvenate!(smc²::SMC²,logprob)
    θ = smc².params

    for i = eachindex(smc².params.x)
        θ.x[i] = metropolis(logprob,smc².chain_len,θ.x[i],smc².mcmc_kernel)
    end
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
        smc².params.logw[i] += logZ(θ.x[i])
    end

    # normalize parameter weights
    normalize!(smc²)

    # resample step
    if ESS(smc².params) < smc².resample_threshold*length(θ)
        smc².mcmc_kernel = random_walk(θ)
        rejuvenate!(smc²,bootstrap_filter)
        reset_weights!(smc².params)
    end
end

# reset after running it T times
function reset!(smc²::SMC²)
    θ = smc².params

    for i = eachindex(θ.xprev)
        θ.xprev[i] = rand(smc².rng,smc².prior)
        θ.x[i] = copy(θ.xprev[i])
    end

    fill!(θ.logw,-log(length(θ)))
    fill!(θ.w,1/length(θ))

    pf.state.t[] = 1
end