export IBIS,expected_parameters,density_tempered,smc²,smc²!

mutable struct IBIS{SSM,XT,ΣT,θT,KT} <: Sampler
    θ::Vector{θT}
    ω::Vector{Float64}

    x::Vector{XT}
    Σ::Vector{ΣT}

    ess::Float64
    ess_min::Float64

    M::Int64
    
    chain::Int64
    logZ::Vector{Float64}

    model::SSM
    prior::Sampleable
    kernel::KT

    acc_threshold::Float64
    acc_ratio::Float64
end

function IBIS(
        M::Int64,
        model::SSM,
        prior::Sampleable,
        chain::Int64,
        ess_threshold::Float64,
        min_ar::Float64 = -1.0
    ) where SSM

    θ = map(m -> rand(prior),1:M)
    ω = (1/M)*ones(Float64,M)

    mods = model.(θ)
    x = [mod.x0 for mod in mods]
    σ = [mod.σ0 for mod in mods]

    logZ = zeros(Float64,M)
    ess = 1.0*M
    ess_min = M*ess_threshold

    XT = eltype(x)
    ΣT = eltype(σ)
    θT = eltype(θ)
    
    # needs testing
    mh_kernel = random_walk_kernel
    KT = typeof(mh_kernel)

    return IBIS{SSM,XT,ΣT,θT,KT}(
        θ,ω,x,σ,ess,ess_min,M,chain,logZ,
        model,prior,mh_kernel,min_ar,0.0
    )
end

function expected_parameters(ibis::IBIS)
    _,ω,_ = reweight(ibis.ω)
    weighted_sample = reduce(hcat,ibis.θ.*ω)
    return sum(weighted_sample,dims=2)
end

function Base.show(io::IO,ibis::IBIS)
    rounded_ess = round(ibis.ess;digits=3)
    expected_θ  = expected_parameters(ibis)
    print(io,"ess     = ",rounded_ess,"\nmean(θ) = ")
    show(io,"text/plain",expected_θ)
end

function resample!(ibis::IBIS)
    a = resample(ibis.ω)

    # reindex the parameter particles
    ibis.θ = ibis.θ[a]
    ibis.ω = ibis.ω[a]

    # reindex the state particles
    ibis.x    = ibis.x[a]
    ibis.Σ    = ibis.Σ[a]
    ibis.logZ = ibis.logZ[a]
end

function rejuvenate!(ibis::IBIS,y::Vector{Float64},ξ::Float64,verbose::Bool)
    acc_array = zeros(Int64,ibis.M)

    # define the PMMH kernel
    pmmh_kernel = ibis.kernel(ibis.θ)
    scales = 0.5*reverse(1:ibis.chain)
    
    if verbose @printf("\t[rejuvenating]") end

    Threads.@threads for m in 1:ibis.M
        for c in 1:ibis.chain
            θ_prop = rand(pmmh_kernel(ibis.θ[m],scales[c]))

            if insupport(ibis.prior,θ_prop)
                x_prop,σ_prop,logZ_prop = log_likelihood(y,ibis.model(θ_prop))

                prior_ratio = logpdf(ibis.prior,θ_prop)-logpdf(ibis.prior,ibis.θ[m])
                likelihood_ratio = ξ*(logZ_prop-ibis.logZ[m])

                log_post_prop = logZ_prop + logpdf(ibis.prior,θ_prop)
                acc_ratio     = likelihood_ratio + prior_ratio

                if (log_post_prop > -Inf && log(rand()) < acc_ratio)
                    ibis.logZ[m] = logZ_prop
                    ibis.θ[m]    = θ_prop
                    ibis.x[m]    = x_prop
                    ibis.Σ[m]    = σ_prop

                    acc_array[m] = 1
                end
            end
        end
        ibis.ω[m] = 1.0
    end

    ibis.acc_ratio = sum(acc_array)/ibis.M
    if verbose @printf("\tacc_rate: %1.5f",ibis.acc_ratio) end

    return ibis
end

rejuvenate!(ibis::IBIS,y::Vector{Float64},verbose::Bool) = rejuvenate!(ibis,y,1.0,verbose)

"""
    smc²(smc,y)

Initialization of Chopin's SMC² at time `t = 1`.
"""
function smc²(ibis::IBIS,y::Vector{Float64})
    for m in 1:ibis.M
        ibis.x[m],ibis.Σ[m],ibis.ω[m] = kalman_filter(
            ibis.model(ibis.θ[m]),
            ibis.x[m],ibis.Σ[m],
            y[1]
        )
    end

    ibis.logZ = ibis.ω
    _,ibis.ω,ibis.ess = reweight(ibis.ω)

    return ibis
end

"""
    smc²!(smc,y)

Online estimation step of Chopin's SMC²
"""
function smc²!(ibis::IBIS,y::Vector{Float64},t::Int64,verbose::Bool=true)
    if verbose @printf("t = %4d\tess = %4.3f",t-1,ibis.ess) end
    
    ## particle degeneration check
    if ibis.ess < ibis.ess_min
        ## resample particles
        resample!(ibis)

        ## particle rejuvenation
        rejuvenate!(ibis,y[1:(t-1)],verbose)
    end

    logω = deepcopy(log.(ibis.ω))
    new_x = similar(ibis.x)
    new_Σ = similar(ibis.Σ)

    ## propagate state particles
    for m in 1:ibis.M
        new_x[m],new_Σ[m],likelihood = kalman_filter(
            ibis.model(ibis.θ[m]),
            ibis.x[m],
            ibis.Σ[m],
            y[t]
        )

        logω[m]      += likelihood
        ibis.logZ[m] += likelihood
    end

    ibis.x = new_x
    ibis.Σ = new_Σ

    ## normalize parameter weights and calculate ESS
    _,ibis.ω,ibis.ess = reweight(logω)
    if verbose print("\n") end
end
