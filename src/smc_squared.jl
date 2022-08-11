export SMC²,update!

#=
SMC² algorithm
=#

struct SMC²{SSM}
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
        θ.w[:] .= 1.0

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

# only defined for univariate valued y_{1:T}
function update!(smc²::SMC²,y::Vector{Float64},verbose::Bool=true)
    # save the log weights at each step of the filter
    log_weights = copy(smc².parameters.w)
    log_weights = log.(log_weights)

    θ  = smc².parameters
    pf = smc².filters
    M  = length(θ)

    ## move filters through one step
    for i in eachindex(smc².parameters.x)
        # this needs a little work...
        smc².likelihoods[i] += update!(pf[i],y[θ.t[]])
        log_weights[i]      += smc².likelihoods[i]
    end

    ## reweight θ-particles
    reweight!(θ,log_weights)
    ess_min = smc².resample_threshold*M

    # if verbose is true print the ess
    if verbose
        @printf(
            "\nt = %4d\tess = %3.5f",
            smc².parameters.t[],
            θ.ess[]
        )
    end

    # in the case of particle degeneration...
    if θ.ess[] < ess_min
        ## resample θ-particles => returns new index θ.a
        resample!(θ)
        acc_rate = 0.0

        # reindex all relevant variables
        θ.x[:] .= θ.x[θ.a]
        pf[:]  .= pf[θ.a]
        smc².likelihoods[:] .= smc².likelihoods[θ.a]

        # define pmmh movement
        catθ = reduce(hcat,θ.x)
        dθ   = (2.38^2) / length(smc².prior)

        # evaluate this to avoid cholesky decomposition errors
        Σ = norm(cov(catθ')) < 1e-12 ? 1e-2*I : dθ*cov(catθ') + 1e-10*I
        if verbose @printf("\t[rejuvenating]") end

        ## particle rejuvenation (could be parallelized)
        for i in eachindex(θ.x)
            for _ in 1:smc².chain_length
                θ_prop    = rand(MvNormal(θ.x[i],Σ))
                pf_prop   = ParticleFilter(smc².rng,smc².N,smc².model(θ_prop))
                logZ_prop = log_likelihood(pf_prop,y[1:θ.t[]])

                prop  = logpdf(smc².prior,θ_prop) + logZ_prop
                old   = logpdf(smc².prior,θ.x[i]) + smc².likelihoods[i]

                #@printf("\n\rprop: %4.5f\told: %4.5f",prop,old)

                if rand() ≤ minimum([1.0,exp(prop-old)])
                    θ.x[i] = θ_prop
                    pf[i]  = pf_prop
                    smc².likelihoods[i] = logZ_prop

                    acc_rate += 1
                end
            end

            # reset θ-weights
            θ.w[i] = 1.0
        end

        acc_rate = acc_rate/(smc².chain_length*M)
        if verbose @printf("\t acc rate: %1.4f",acc_rate) end
    end

    smc².parameters.t[] += 1
end
