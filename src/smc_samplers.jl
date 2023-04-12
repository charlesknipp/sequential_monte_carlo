export SMC,expected_parameters,density_tempered,smc²,smc²!

abstract type Sampler end

mutable struct SMC{SSM,XT,θT,KT} <: Sampler
    θ::Vector{θT}
    ω::Vector{Float64}

    x::Vector{Vector{XT}}
    w::Vector{Vector{Float64}}

    ess::Float64
    ess_min::Float64

    N::Int64
    M::Int64
    
    chain::Int64
    logZ::Vector{Float64}

    model::SSM
    prior::Sampleable
    kernel::KT

    acc_threshold::Float64
    acc_ratio::Float64
end

function SMC(
        N::Int64,M::Int64,
        model::SSM,
        prior::Sampleable,
        chain::Int64,
        ess_threshold::Float64,
        min_ar::Float64 = -1.0
    ) where SSM

    θ = map(m -> rand(prior),1:M)
    ω = (1/M)*ones(Float64,M)

    x = fill(preallocate(model(θ[1]),N),M)
    w = fill(zeros(Float64,N),M)

    logZ = zeros(Float64,M)
    ess = 1.0*M
    ess_min = M*ess_threshold

    XT = eltype(x[1])
    θT = eltype(θ)
    
    # needs testing
    mh_kernel = random_walk_kernel
    KT = typeof(mh_kernel)

    return SMC{SSM,XT,θT,KT}(
        θ,ω,x,w,ess,ess_min,N,M,chain,logZ,
        model,prior,mh_kernel,min_ar,0.0
    )
end

function expected_parameters(smc::SMC)
    _,ω,_ = reweight(smc.ω)
    weighted_sample = reduce(hcat,smc.θ.*ω)
    return sum(weighted_sample,dims=2)
end

function Base.show(io::IO,smc::SMC)
    rounded_ess = round(smc.ess;digits=3)
    expected_θ  = expected_parameters(smc)
    print(io,"ess     = ",rounded_ess,"\nmean(θ) = ")
    show(io,"text/plain",expected_θ)
end

function resample!(smc::SMC)
    a = resample(smc.ω)

    # reindex the parameter particles
    smc.θ = smc.θ[a]
    smc.ω = smc.ω[a]

    # reindex the state particles
    smc.x    = smc.x[a]
    smc.logZ = smc.logZ[a]
end

# define for univariate θ
function random_walk_kernel(θ::Vector{Float64})
    dθ = 2.83^2
    σ  = norm(cov(θ)) < 1.e-8 ? 1.e-2 : dθ*cov(θ) + 1.e-10
    
    return (x,scale) -> Normal(x,scale*σ)
end

# define for multivariate θ (should be typed differently)
function random_walk_kernel(θ::Vector{Vector{Float64}})
    θ  = hcat(θ...)
    dθ = 2.83^2 / size(θ,1)
    Σ  = norm(cov(θ')) < 1.e-8 ? 1.e-2I : dθ*cov(θ') + 1.e-10I

    return (x,scale) -> MvNormal(x,scale*Σ)
end

function rejuvenate!(smc::SMC,y::Vector{Float64},ξ::Float64,verbose::Bool)
    acc_array = zeros(Int64,smc.M)

    # define the PMMH kernel
    pmmh_kernel = smc.kernel(smc.θ)
    scales = 0.5*reverse(1:smc.chain)
    
    if verbose @printf("\t[rejuvenating]") end

    Threads.@threads for m in 1:smc.M
        for c in 1:smc.chain
            θ_prop = rand(pmmh_kernel(smc.θ[m],scales[c]))

            if insupport(smc.prior,θ_prop)
                x_prop,w_prop,logZ_prop = log_likelihood(
                    smc.N,
                    y,
                    smc.model(θ_prop)
                )

                prior_ratio = logpdf(smc.prior,θ_prop)-logpdf(smc.prior,smc.θ[m])
                likelihood_ratio = ξ*(logZ_prop-smc.logZ[m])

                log_post_prop = logZ_prop + logpdf(smc.prior,θ_prop)
                acc_ratio     = likelihood_ratio + prior_ratio

                if (log_post_prop > -Inf && log(rand()) < acc_ratio)
                    smc.logZ[m] = logZ_prop
                    smc.θ[m]    = θ_prop
                    smc.x[m]    = x_prop
                    smc.w[m]    = w_prop

                    acc_array[m] = 1
                end
            end
        end
        smc.ω[m] = 1.0
    end

    smc.acc_ratio = sum(acc_array)/smc.M
    if verbose @printf("\tacc_rate: %1.5f",smc.acc_ratio) end

    return smc
end

rejuvenate!(smc::SMC,y::Vector{Float64},verbose::Bool) = rejuvenate!(smc,y,1.0,verbose)

"""
    exchange!(smc,y)

The exchange step checks whether the acceptance ratio is sufficiently small in
which case it will double the number of state particles. This specific scheme
is taken from (Chopin 2013) and closely follows his Python implementation.

More precisely, if the acceptance ratio is less than some threshold...
    - double N
    - run particle filters over a new set of state particles
    - get the new likelihoods from 1:t
    - reset the log weights as the difference between likelihoods
"""
function exchange!(smc::SMC,y::Vector{Float64},verbose::Bool)
    if smc.acc_ratio < smc.acc_threshold
        ## double the number of state particles
        if smc.N <= 4096
            smc.N *= 2
            if verbose @printf("\t%d particles added",smc.N) end

            # not sure if I need to reallocate smc.x and smc.w
            new_logZ = zeros(Float64,smc.M)

            ## generate new set of particles
            Threads.@threads for m in 1:smc.M
                smc.x[m],smc.w[m],new_logZ[m] = log_likelihood(
                    smc.N,
                    y,
                    smc.model(smc.θ[m])
                )
            end

            ## normalize the weights and calculate the ESS
            _,smc.ω,smc.ess = reweight(new_logZ.-smc.logZ)
            smc.logZ = new_logZ
        else
            print("\n\t[cannot exceed max state particles]")
        end
    end
end


"""
    density_tempered(smc,y)

Implementation of Duan & Fulop's density tempered particle filter.

```julia
> mod_func(θ) = StateSpaceModel(
    LinearGaussian(θ[1],1.0,θ[2],θ[3],0.0),
    (1,1)
  )
> prior = product_distribution([
    TruncatedNormal(0,1,-1,1),
    LogNormal(),
    LogNormal()
  ])
> dt_smc = SMC(512,1024,mod_func,prior,3,0.5)
> density_tempered(dt_smc,y)
  ξ = 0.00825     ess = 255.986   [rejuvenating]  acc_rate: 0.18594
  ξ = 0.03895     ess = 256.000   [rejuvenating]  acc_rate: 0.21055
  ξ = 0.11587     ess = 255.998   [rejuvenating]  acc_rate: 0.16055
  ξ = 0.27741     ess = 255.999   [rejuvenating]  acc_rate: 0.17656
  ξ = 0.67719     ess = 256.000   [rejuvenating]  acc_rate: 0.15664
  ξ = 1.00000     ess = 415.016
> expected_parameters(dt_smc)
  3×1 Matrix{Float64}:
   0.503320909344024
   1.024557844205593
   0.9752674712290297
```
"""
function density_tempered(smc::SMC,y::Vector{Float64},verbose=true)
    Threads.@threads for m in 1:smc.M
        smc.x[m],smc.w[m],smc.logZ[m] = log_likelihood(
            smc.N,
            y,
            smc.model(smc.θ[m])
        )
    end

    ## normalize the weights and calculate the ESS
    _,smc.ω,smc.ess = reweight(smc.logZ)

    ξ = 0.0
    while ξ < 1.0
        # force resample if ξ < 1.0
        resample_flag = true

        ## (2.2.2) find optimal ξ and reweight particles
        lower_bound = oldξ = ξ
        upper_bound = 2.0

        # bisection search for ξ
        local newξ,logω
        while upper_bound-lower_bound > 1.e-6
            newξ = (upper_bound+lower_bound)/2.0
            logω = (newξ-oldξ)*smc.logZ

            _,smc.ω,smc.ess = reweight(logω)

            if smc.ess == smc.ess_min
                break
            elseif smc.ess < smc.ess_min
                upper_bound = newξ
            else
                lower_bound = newξ
            end
        end

        # account for corner solutions
        if newξ ≥ 1.0
            resample_flag = false
            newξ = 1.0
            logω = (newξ-oldξ)*smc.logZ
            _,smc.ω,smc.ess = reweight(logω)
        end

        ξ = newξ

        if verbose @printf("ξ = %1.5f\tess = %4.3f",ξ,smc.ess) end

        if resample_flag
            ## (2.2.3) resample particles
            resample!(smc)

            ## (2.2.4) moving the particles
            rejuvenate!(smc,y,ξ,verbose)
        end
        if verbose print("\n") end
    end
end

"""
    smc²(smc,y)

Initialization of Chopin's SMC² at time `t = 1`.
"""
function smc²(smc::SMC,y::Vector{Float64})
    for m in 1:smc.M
        smc.x[m],smc.w[m],smc.ω[m] = bootstrap_filter(
            smc.N,
            y[1],
            smc.model(smc.θ[m])
        )
    end

    smc.logZ = smc.ω
    _,smc.ω,smc.ess = reweight(smc.ω)

    return smc
end

"""
    smc²!(smc,y)

Online estimation step of Chopin's SMC²
"""
function smc²!(smc::SMC,y::Vector{Float64},t::Int64,verbose::Bool=true)
    if verbose @printf("t = %4d\tess = %4.3f",t-1,smc.ess) end
    
    ## particle degeneration check
    if smc.ess < smc.ess_min
        ## resample particles
        resample!(smc)

        ## particle rejuvenation
        rejuvenate!(smc,y[1:(t-1)],verbose)

        ## exchange particles if acceptance ratio is sufficiently low
        exchange!(smc,y[1:(t-1)],verbose)
    end

    ## propagate state particles
    logω = deepcopy(log.(smc.ω))
    for m in 1:smc.M
        likelihood,smc.w[m] = bootstrap_filter!(
            smc.x[m],
            smc.w[m],
            y[t],
            smc.model(smc.θ[m])
        )

        logω[m]     += likelihood
        smc.logZ[m] += likelihood
    end

    ## normalize parameter weights and calculate ESS
    _,smc.ω,smc.ess = reweight(logω)
    if verbose print("\n") end
end
