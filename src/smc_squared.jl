export SMC²,update!

#=
SMC² algorithm
=#

mutable struct SMC²{SSM}
    parameters::Particles{Vector{Float64}}
    filters::Vector{ParticleFilter}
    likelihoods::Vector{Float64}

    prior::Sampleable
    model::SSM

    N::Int64
    chain_length::Int64

    resample_threshold::Float64
    rng::AbstractRNG

    function SMC²(
            N::Int64,M::Int64,
            prior::Sampleable,
            model::SSM,
            chain_length::Int64,
            resample_threshold::Float64,
            rng::AbstractRNG
        ) where SSM

        θ  = Particles{Vector{Float64}}([rand(rng,prior) for _ in 1:M])
        pf = [ParticleFilter(rng,N,model(θ.x[i])) for i in 1:M]
        #θ.w[:] .= 1.0

        new{typeof(model)}(
            θ,
            pf,
            zeros(Float64,M),
            prior,
            model,
            N,
            chain_length,
            resample_threshold,
            rng
        )
    end
end

#=
BUG: in rejuvenation, the time index of each state particle deviates from the
given index by the parameter vector.

Interesetingly the problem is not with pf_prop, but when it is placed in the
array of filters
=#

# only defined for univariate valued y_{1:T}
function update!(
        smc²::SMC²,
        y::Vector{Float64},
        verbose::Bool=true,
        debug::Bool=false
    )

    # save the log weights at each step of the filter
    log_weights = copy(smc².parameters.w)
    log_weights = log.(log_weights)

    θ  = smc².parameters
    pf = smc².filters
    M  = length(θ)

    ## move filters through one step
    for i in eachindex(smc².parameters.x)
        # this needs a little work...
        marginal_likelihood  = update!(pf[i],y[θ.t[]])
        smc².likelihoods[i] += marginal_likelihood
        log_weights[i]      += marginal_likelihood
    end

    ## reweight θ-particles
    reweight!(θ,log_weights)
    ess_min = smc².resample_threshold*M

    if verbose
        @printf(
            "\nt = %3d   ess = %3.5f",
            smc².parameters.t[],
            θ.ess[]
        )
    end

    # in the case of particle degeneration...
    if θ.ess[] < ess_min
        ## resample θ-particles => returns new index θ.a
        resample!(θ)
        acc_rate = 0.0

        # reindex all relevant variables (maybe problematic)
        θ.x[:] .= θ.x[θ.a]
        pf      = pf[θ.a]
        smc².likelihoods = smc².likelihoods[θ.a]

        # define pmmh movement
        catθ = reduce(hcat,θ.x)
        dθ   = (2.38^2) / length(smc².prior)

        # evaluate this to avoid cholesky decomposition errors
        Σ = norm(cov(catθ')) < 1e-12 ? 1e-2*I : dθ*cov(catθ') + 1e-10*I
        if verbose @printf("\t[rejuvenating]") end

        ## particle rejuvenation (could be parallelized)
        for i in eachindex(θ.x)
            for _ in 1:smc².chain_length
                θ_prop = rand(MvNormal(θ.x[i],Σ))

                # proposed particle must be in the support of the prior
                if insupport(smc².prior,θ_prop)
                    local pf_prop   = ParticleFilter(smc².rng,smc².N,smc².model(θ_prop))
                    local logZ_prop = log_likelihood(pf_prop,y[1:θ.t[]])

                    prior_ratio = logpdf(smc².prior,θ_prop)-logpdf(smc².prior,θ.x[i])
                    likelihood_ratio = logZ_prop-smc².likelihoods[i]

                    log_post_prop = logZ_prop + logpdf(smc².prior,θ_prop)
                    acc_ratio     = likelihood_ratio + prior_ratio

                    if (log_post_prop > -Inf && log(rand()) < acc_ratio)
                        θ.x[i] = θ_prop
                        pf[i]  = pf_prop
                        smc².likelihoods[i] = logZ_prop

                        acc_rate += 1
                    end
                end
            end

            # reset θ-weights
            θ.w[i] = 1.0
        end

        acc_rate = acc_rate/(smc².chain_length*M)
        if verbose @printf("\tacc rate: %1.4f",acc_rate) end
    end

    if debug
        x_t = mean([filter.state.t[] for filter in pf])
        θ_t = smc².parameters.t[]
    
        @printf("\n   x.t = %3d   θ.t = %3d",x_t,θ_t)
    end

    smc².parameters.t[] += 1
end
