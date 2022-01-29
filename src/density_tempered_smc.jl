export densityTemperedSMC

function gridSearch(ξ::Float64,θ::Particles,B::Float64)
    # solve for such that θ.ess == B*M
    M = length(θ)
    f(Δ::Float64) = ESS(Δ*θ.logw) - B*M

    if f(1.0-ξ) > 0
        δ = 1.0 - ξ
        newξ = 1.0
    else
        δ = secant(f,1.e-12,1-ξ)
        newξ = ξ + δ
    end

    return newξ
end

# FIX: will sometimes result in bad covariance matrices
function randomWalk(θ::Particles,kernel::Function,c::Float64=0.5)
    M = length(θ.x)
    x = reduce(hcat,θ.x)

    # calculate the weighted mean and covariance
    μ = vec(mean(x,weights(θ.w),2))
    Σ = cov(x,weights(θ.w),2)

    # finish this to generate a particle set
    pθ = kernel(μ,c*Σ)
    newθ = rand(pθ,M)

    return Particles([newθ[:,m] for m in 1:M]),pθ
end

# this is totally untested, but the logic is 100% there
function randomWalkMH(
        θ::Particles,prior::Function,model,N::Int64,y::Vector{Float64},pθ::Sampleable;
        B::Float64=0.5,c::Float64=0.5,len_chain::Int64=5
    )

    M = length(θ)

    for _ in 1:len_chain
        newθ = randomWalk(θ,prior,c)

        # can be parallelized
        for m in 1:M
            newΘm = StateSpaceModel(model(newθ[m]...))
            newXm = bootstrapFilter(N,y,newΘm,B)
            newθ.logw[m] = sum([Xmt.logμ for Xmt in newXm.p])

            αm = ξ*(newθ.logw[m]-θ.logw[m])
            αm += (logpdf(pθ,θ)-logpdf(pθ,newθ))

            if rand() ≤ minimum([αm,1.0])
                θ.x = newθ
                θ.logw = 0.0
            end
        end
    end

    # re-normalize everything
    return Particles(θ.x,θ.logw)
end

function densityTemperedSMC(
        N::Int64,
        M::Int64,
        y::Vector{Float64},
        θ0::Vector{Float64},
        prior::Function,
        B::Float64 = 0.5,
        model = LinearGaussian
    )
    
    # initialize particles
    pθ = prior(θ0,Matrix{Float64}(I,k,k))
    θ = rand(pθ,M)
    θ = Particles([θ[:,m] for m in 1:M])

    # parallelize this
    for m in 1:M
        Θm = StateSpaceModel(model(θ.x[m]...))
        Xm = bootstrapFilter(N,y,Θm,0.5)
        θ.logw[m] = sum([Xmt.logμ for Xmt in Xm.p])
    end

    while ξ ≤ 1.0
        newξ = gridSearch(ξ,θ,B)
        θ = reweight(θ,[(newξ-ξ)*θ.logw[m] for m in 1:M])

        if θ.ess < B*M
            θ = randomWalkMH(θ,prior,model,N,y,pθ)
        end
    end

    return θ
end