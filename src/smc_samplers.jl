export SMC,expected_parameters,density_tempered,smc²,smc²!

mutable struct SMC{SSM}
    θ::Vector{Vector{Float64}}
    ω::Vector{Float64}

    x::Vector{Vector{Float64}}
    w::Vector{Vector{Float64}}

    ess::Float64
    ess_min::Float64

    N::Int64
    M::Int64
    
    chain::Int64
    logZ::Vector{Float64}

    model::SSM
    prior::Sampleable

    rng::AbstractRNG
end

function SMC(
        rng::AbstractRNG,
        N::Int64,M::Int64,
        model::SSM,
        prior::Sampleable,
        chain::Int64,
        ess_threshold::Float64
    ) where SSM

    θ = map(m -> rand(rng,prior),1:M)
    ω = (1/M)*ones(Float64,M)

    x = fill(zeros(Float64,N),M)
    w = similar(x)

    logZ = zeros(Float64,M)
    ess = 1.0*M
    ess_min = M*ess_threshold

    return SMC{typeof(model)}(θ,ω,x,w,ess,ess_min,N,M,chain,logZ,model,prior,rng)
end

function expected_parameters(smc::SMC)
    _,ω,_ = normalize(smc.ω)
    weighted_sample = reduce(hcat,smc.θ.*ω)
    return sum(weighted_sample,dims=2)
end

function resample!(smc::SMC)
    a = resample(smc.rng,smc.ω)

    # reindex the parameter particles
    smc.θ = smc.θ[a]
    smc.ω = smc.ω[a]

    # reindex the state particles
    smc.x    = smc.x[a]
    smc.logZ = smc.logZ[a]
end

function rejuvenate!(smc::SMC,y::Vector{Float64},ξ::Float64,verbose::Bool)
    acc_rate = 0.0

    # define the PMMH kernel
    catθ = reduce(hcat,smc.θ)
    dθ   = (2.83^2)/length(smc.prior)

    Σ = norm(cov(catθ')) < 1.e-12 ? 1.e-2*I : dθ*cov(catθ') + 1.e-10I
    if verbose @printf("\t[rejuvenating]") end

    Threads.@threads for m in 1:smc.M
        for _ in 1:smc.chain
            # currently only supports random walk
            θ_prop = rand(smc.rng,MvNormal(smc.θ[m],Σ))

            if insupport(smc.prior,θ_prop)
                x_prop,w_prop,logZ_prop = log_likelihood(
                    smc.rng,
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

                    acc_rate += 1
                end
            end
        end
        smc.ω[m] = 1.0
    end

    acc_rate = acc_rate/(smc.chain*smc.M)
    if verbose @printf("\tacc_rate: %1.5f",acc_rate) end

    return smc
end

rejuvenate!(smc::SMC,y::Vector{Float64},verbose::Bool) = rejuvenate!(smc,y,1.0,verbose)

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
            smc.rng,
            smc.N,
            y,
            smc.model(smc.θ[m])
        )
    end

    ## normalize the weights and calculate the ESS
    _,smc.ω,smc.ess = normalize(smc.logZ)

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

            _,smc.ω,smc.ess = normalize(logω)

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
            _,smc.ω,smc.ess = normalize(logω)
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
        smc.x[m],smc.w[m],smc.ω[m] = particle_filter(
            smc.rng,
            smc.N,
            y[1],
            smc.model(smc.θ[m]),
            nothing
        )
    end

    smc.logZ = smc.ω
    _,smc.ω,smc.ess = normalize(smc.ω)

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

        ## rejuvenation step
        rejuvenate!(smc,y[1:(t-1)],verbose)
    end

    logω = deepcopy(log.(smc.ω))
    for m in 1:smc.M
        likelihood,smc.w[m],_ = particle_filter!(
            smc.rng,
            smc.x[m],
            smc.w[m],
            y[t],
            smc.model(smc.θ[m]),
            nothing
        )

        logω[m]     += likelihood
        smc.logZ[m] += likelihood
    end

    ## normalize parameter weights and calculate ESS
    _,smc.ω,smc.ess = normalize(logω)
    if verbose print("\n") end
end