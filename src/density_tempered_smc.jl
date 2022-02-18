export densityTemperedSMC

mutable struct DensityTemperedSMC
    M::Int64
    N::Int64

    iteration::Int64
    y::AbstractVector

    prior::Sampleable
    model::Function

    ξ::Float64
    B::Float64
    logZ::Vector{Float64}

    # particle characteristics
    θ::Vector{Vector{Float64}}
    logw::Vector{Float64}
    w::Vector{Float64}
    ess::Float64
end

function reweight!(smc::DensityTemperedSMC)
    smc.iteration += 1

    lower_bound = oldξ = smc.ξ
    upper_bound = 2.0

    local ess,newξ,logw

    while upper_bound-lower_bound > 1.e-6
        newξ = (upper_bound+lower_bound)/2.0

        logw = (newξ-oldξ)*smc.logZ
        logw = logw.-maximum(logw)

        w = exp.(logw)
        w = w/sum(w)

        ess = 1.0/sum(w.^2)

        if ess == smc.B
            break
        elseif ess < smc.B
            upper_bound = newξ
        else
            lower_bound = newξ
        end
    end

    if newξ ≥ 1.0
        newξ = 1.0

        logw = (newξ-oldξ)*smc.logZ
        logw = logw.-maximum(logw)
    end

    smc.ess = ess
    smc.ξ = newξ

    # display the tempering sequence in the console
    @printf("ξ[%0d] = %0.6f\u1b[K\n",smc.iteration,smc.ξ)

    # take the log of the normalized weights to reduce numerical problems
    w = exp.(logw)
    smc.w = w/sum(w)
    smc.logw = log.(smc.w)
end

function resample!(smc::DensityTemperedSMC)
    a = wsample(1:smc.M,smc.w,smc.M)

    smc.θ = smc.θ[a]
    smc.logZ = smc.logZ[a]
end

function rejuvinate!(smc::DensityTemperedSMC,c::Float64,len_chain::Int64=5)
    print("rejuvenating...")
    θ = reduce(hcat,smc.θ)

    # calculate the weighted mean and covariance
    μ = vec(mean(θ,weights(smc.w),2))
    Σ = cov(θ,weights(smc.w),2)

    # kinda sussy
    newθ = rand(MvNormal(μ,c*Σ),smc.M)
    newθ[3:4,:] = abs.(newθ[3:4,:])

    for _ in 1:len_chain
        u = rand(smc.M)
        for m in 1:smc.M
            # TODO: implement new SSM behavior
            Θm = StateSpaceModel(smc.model(newθ[:,m]...))
            newXm = bootstrapFilter(smc.N,smc.y,Θm)
            logZm = sum([Xmt.logμ for Xmt in newXm.p])

            # TODO: store logpdf(prior,θ) to avoid redundancies
            αm = smc.ξ*(logZm-smc.logZ[m])
            αm += (logpdf(smc.prior,smc.θ[m])-logpdf(smc.prior,newθ[:,m]))
            αm = exp(αm)

            if u[m] ≤ minimum([αm,1.0])
                smc.θ[m] = newθ[:,m]
                smc.logZ[m] = logZm
            end
        end
    end

    # reset the cursor to print ξ to stdout
    print("\r")
end

# TODO: add [rng=GLOBAL_RNG] to aid in testing
function densityTemperedSMC(
        M::Int64,
        N::Int64,
        y::AbstractVector,
        prior::Function,
        θ0::Vector{Float64},
        model::Function,
        threshold::Float64 = 0.5
    )
    iter = 0

    # consider restructuring θ
    pθ = prior(θ0,Matrix{Float64}(I(length(θ0))))
    θ  = rand(pθ,M)
    θ  = [θ[:,m] for m in 1:M]

    # consider rewriting this method
    logZ = zeros(Float64,M)
    for m in 1:M
        Θm = StateSpaceModel(model(θ[m]...))
        Xm = bootstrapFilter(N,y,Θm)
        logZ[m] = sum([Xmt.logμ for Xmt in Xm.p])
    end

    # initialize weights at 1/M
    logw = fill(-1*log(M),M)
    w    = fill(1/M,M)
    ess  = M

    ξ0 = 1.e-12
    B  = threshold*M

    # initialize the algorithm
    smc = DensityTemperedSMC(M,N,iter,y,pθ,model,ξ0,B,logZ,θ,logw,w,ess)

    while smc.ξ < 1.0
        reweight!(smc)

        if smc.ess < B
            resample!(smc)
            rejuvinate!(smc,0.5)
        end
    end

    return smc.θ,smc.w
end