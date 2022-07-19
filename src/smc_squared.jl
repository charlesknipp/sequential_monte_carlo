export SMC²,reset!,random_walk,metropolis,rejuvenate!,update_importance!,reset!

#=
    SMC² is still a work in progress, but it is well on its way to completion.
    The current object keeps track of the particle objects and uses a function
    defined at each iteration to calculate the likelihood for each θ particle.

    The main iteration is performed via update_importance!(...) which is an
    in-place function to update the parameters of an SMC² object. Speaking of,
    SMC² objects track parameter particles and their associated particle fil-
    ters; this is what allows the algorithm to really fly in terms of speed.

    This still needs some work, specifically on the MCMC aspect; acceptance is
    far too low and declines exponentially with time. The causes are yet to be
    identified, but a solution will present itself in due time.
=#

struct SMC²{ΘT,XT,PΘ,SSM}
    params::ΘT
    filters::XT
    likelihoods::Vector{Float64}

    prior::PΘ
    model::SSM

    N::Int
    chain_len::Int

    resample_threshold::Float64
    rng::AbstractRNG
end

# default constructor, which establishes the algorithm at t=0
function SMC²(M::Int,N::Int,prior,model,B,chain_len,rng=Random.GLOBAL_RNG)
    # establish parameter particles
    mod_type = eltype(prior)
    dimθ = length(prior)

    θprev = Vector{SVector{dimθ,mod_type}}([rand(rng,prior) for m=1:M])
    θ = deepcopy(θprev)

    logw = fill(0.0,M)
    w = fill(1.0,M)

    θ = Particles(θ,θprev,logw,w,Ref(0.),collect(1:M),Ref(1))
    
    # for each parameter particle establish state particles
    pf = [ParticleFilter(N,model(θ.x[i]),Inf,rng) for i in 1:M]

    return SMC²(θ,pf,zeros(Float64,M),prior,model,N,chain_len,B,rng)
end

# random walk kernel using the covariance of the particle set
function random_walk(θ0)
    x  = reduce(hcat,θ0.x)
    dθ = (2.38)^2 / length(θ0.x[1])

    # whacky covariance is for numerical stability
    Σ = norm(cov(x')) < 1e-12 ? 1e-2*I : dθ*cov(x') + 1e-10*I

    # returns a function that takes a vector θ
    return θ -> MvNormal(θ,Σ)
end

# naive random walk which just uses an identity matrix as covariance
function naive_random_walk(θ0)
    dθ = (2.38)^2 / sqrt(length(θ0.x[1]))

    # returns a function that takes a vector θ
    return θ -> MvNormal(θ,dθ*I(length(θ0)))
end

# resample parameter particles
function resample!(smc²::SMC²)
    θ = smc².params
    a = wsample(smc².rng,1:length(θ),θ.w,length(θ))

    filters = deepcopy(smc².filters)
    probabilities = deepcopy(smc².likelihoods)

    # this does not reindex the weights!!!
    for i = eachindex(θ.x)
        smc².params.x[i]    = θ.xprev[a[i]]
        smc².filters[i]     = filters[a[i]]
        smc².likelihoods[i] = probabilities[a[i]]
    end

    ## won't work since Particles are immutable
    #  smc².params.w = smc².params.w[a]

    #smc².filters     .= smc².filters[a]
    #smc².likelihoods .= smc².likelihoods[a]

    return smc².params.x,smc².filters,smc².likelihoods
end

# rejuvenation by way of metropolis hastings
function rejuvenate!(smc²::SMC²,y)
    acc_rate = 0
    mcmc_kernel = random_walk(smc².params)

    # function to construct a pf given vector of parameters
    filter_params(θ) = ParticleFilter(smc².N,smc².model(θ),Inf,smc².rng)

    # not sure if this writes globally to smc²
    for i = eachindex(smc².params.x)
        θ   = smc².params.x[i]
        ll  = smc².likelihoods[i]
        pf  = smc².filters[i]
        
        for _ in 1:smc².chain_len
            # get the log likelihood at time t for the newly proposed θ
            proposal_θ  = rand(smc².rng,mcmc_kernel(θ))
            proposal_pf = filter_params(proposal_θ)
            proposal_ll = log_likelihood(proposal_pf,y[1:smc².params.t[]])

            # add the logpdf of prior to the likelihood to get the ratio
            old_likelihood = logpdf(smc².prior,θ) + ll
            new_likelihood = logpdf(smc².prior,proposal_θ) + proposal_ll

            #@printf("\n[%2d] old prob: %5.3f\tnew prob: %5.3f",i,ll,proposal_ll)

            if log(rand(smc².rng)) ≤ minimum([1,new_likelihood-old_likelihood])
                θ  = proposal_θ
                ll = proposal_ll
                pf = proposal_pf
                acc_rate += 1

                #print("\t[accepted]")
            end
        end
    end

    @printf("\t acc rate: %1.4f",acc_rate/(length(smc².params.x)*smc².chain_len))

    return smc².params.x
end

function reset_weights!(smc²::SMC²)
    fill!(smc².params.logw,0.0)
    fill!(smc².params.w,1.0)
    smc².params.maxw[] = 0.0
end

@inline normalize!(smc²::SMC²) = normalize!(smc².params)

# this is defines a single step t of the algorithm
function update_importance!(smc²::SMC²,y)
    # weigh each θ particle
    for i in eachindex(smc².params.x)
        ## not sure if I weighted this properly
        smc².likelihoods[i] += update!(smc².filters[i],y[smc².params.t[]])
        smc².params.logw[i] += smc².likelihoods[i]
    end

    # normalize θ weights and calculate the ESS
    normalize!(smc²)
    ess = sum(smc².params.w)^2 / sum(abs2,smc².params.w)

    # resample step
    if ess < smc².resample_threshold*length(smc².params.x)
        # print progress
        @printf("\nt = %4d\tess = %3.5f\t[rejuvenating]",smc².params.t[],ess)

        resample!(smc²)     # resample filters, particles, and likelihoods
        rejuvenate!(smc²,y) # rejuvenate with random walk MH kernel

        reset_weights!(smc²)
    else
        # print progress
        @printf("\nt = %4d\tess = %3.5f",smc².params.t[],ess)
    end

    # TODO: adaptive resampling?

    # propogate the parameter particles
    copyto!(smc².params.xprev,smc².params.x)
    smc².params.t[] += 1
end

# reset after running it T times
function reset!(smc²::SMC²)
    θ  = smc².params
    l  = smc².likelihoods
    pf = smc².filters

    # resample from the prior and copy
    for i = eachindex(θ.xprev)
        θ.xprev[i] = rand(smc².rng,smc².prior)
        θ.x[i] = copy(θ.xprev[i])

        # reset the particle filters using new values
        pf = ParticleFilter(smc².N,smc².model(θ.x[i]),Inf,smc².rng)
    end

    # reset weights and likelihoods
    fill!(θ.logw,0.0)
    fill!(θ.w,1.0)
    fill!(l,0.0)

    smc².params.t[] = 1
end