export density_tempered_pf

# only for vectors of Float64
function density_tempered_pf(
        N::Int64,
        M::Int64,
        y::Vector{Float64},
        prior::Sampleable,
        model::SSM,
        chain_length::Int64,
        ess_threshold::Float64,
        rng::AbstractRNG,
        verbose::Bool=true
    ) where SSM

    #logZ = zeros(Float64,M)
    θ  = Particles{Vector{Float64}}([rand(rng,prior) for _ in 1:M])
    pf = [ParticleFilter(rng,N,model(θ.x[i])) for i in 1:M]

    # change to pmap(x -> log_likelihood(x,y),pf)
    logZ = map(x -> log_likelihood(x,y),pf)
    reweight!(θ,logZ)

    ξ = 0.0

    # define the main loop for iteration
    while ξ < 1.0
        lower_bound = oldξ = ξ
        upper_bound = 2.0

        # bisection search for ξ
        local newξ,logw
        while upper_bound-lower_bound > 1.e-6
            newξ = (upper_bound+lower_bound)/2.0
            logw = (newξ-oldξ)*logZ
            
            reweight!(θ,logw)

            if θ.ess[] == ess_threshold*M
                break
            elseif θ.ess[] < ess_threshold*M
                upper_bound = newξ
            else
                lower_bound = newξ
            end
        end

        # make sure to account for corner solutions
        if newξ ≥ 1.0
            newξ = 1.0
    
            logw = (newξ-oldξ)*logZ
            reweight!(θ,logw)
        end

        ξ = newξ

        # if verbose is true print the ess
        if verbose @printf("\nξ = %2.5f\tess = %3.5f",ξ,θ.ess[]) end

        # if ess is small enough, then rejuvenate
        if θ.ess[] < ess_threshold*M
            resample!(θ)
            acc_rate = 0.0

            #reindex particles, filters and likelihoods
            θ.x[:]  .= θ.x[θ.a]
            pf[:]   .= pf[θ.a]
            logZ[:] .= logZ[θ.a]

            # define pmmh movement
            catθ = reduce(hcat,θ.x)
            dθ   = (2.38^2) / length(prior)

            # evaluate this to avoid cholesky decomposition errors
            Σ = norm(cov(catθ')) < 1e-12 ? 1e-2*I : dθ*cov(catθ') + 1e-10*I
            if verbose @printf("\t[rejuvenating]") end

            ## particle rejuvenation (could be parallelized)
            for i in eachindex(θ.x)
                for _ in 1:chain_length
                    θ_prop = rand(MvNormal(θ.x[i],Σ))

                    # proposed particle must be in the support of the prior
                    if insupport(prior,θ_prop)
                        pf_prop   = ParticleFilter(rng,N,model(θ_prop))
                        logZ_prop = log_likelihood(pf_prop,y)

                        prior_ratio = logpdf(prior,θ_prop)-logpdf(prior,θ.x[i])
                        likelihood_ratio = ξ*(logZ_prop-logZ[i])

                        log_post_prop = logZ_prop + logpdf(prior,θ_prop)
                        acc_ratio     = likelihood_ratio + prior_ratio

                        if (log_post_prop > -Inf && log(rand()) < acc_ratio)
                            θ.x[i]  = θ_prop
                            pf[i]   = pf_prop
                            logZ[i] = logZ_prop

                            acc_rate += 1
                        end
                    end
                end

                # reset θ-weights
                θ.w[i] = 1.0
            end

            acc_rate = acc_rate/(chain_length*M)
            if verbose @printf("\t acc rate: %1.4f",acc_rate) end
        end
    end
    
    if verbose @printf("\n") end

    return θ
end
