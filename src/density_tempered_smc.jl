export densityTemperedSMC

function gridSearch(ξ::Float64,θ::Particles,B::Float64)
    # solve for such that θ.ess == B*M
    M = length(θ)

    f(Δ::Float64) = ESS(Δ*θ.logw) - B*M
    δ = bisection(f,1.e-12,1-ξ)

    return ξ+δ
end

# FIX: will sometimes result in bad covariance matrices
function randomWalk(θ::Particles,c::Float64=0.5)
    M = length(θ.x)
    x = reduce(hcat,θ.x)

    # calculate the weighted mean and covariance
    μ = vec(mean(x,weights(θ.w),2))
    Σ = cov(x,weights(θ.w),2)

    # finish this to generate a particle set
    newθ = rand(MvNormal(μ,c*Σ),M)

    # this is wrong, but present for testing purposes
    newθ[3:4,:] = abs.(newθ[3:4,:])

    return Particles([newθ[:,m] for m in 1:M])
end

# this is totally untested, but the logic is 100% there
function randomWalkMH(
        θ::Particles,model,N::Int64,y::Vector{Float64},pθ::Sampleable,ξ::Float64;
        B::Float64=0.5,c::Float64=0.5,len_chain::Int64=3
    )

    M = length(θ)
    acc = zeros(Int64,M)

    for _ in 1:len_chain
        newθ = randomWalk(θ,c)

        # can be parallelized
        for m in 1:M
            newΘm = StateSpaceModel(model(newθ.x[m]...))
            newXm = bootstrapFilter(N,y,newΘm,B)
            newθ.logw[m] = sum([Xmt.logμ for Xmt in newXm.p])

            αm = ξ*newθ.logw[m]-θ.logw[m]
            αm += (logpdf(pθ,θ.x[m])-logpdf(pθ,newθ.x[m]))

            if rand() ≤ minimum([αm,1.0])
                θ.x[m] = newθ.x[m]

                # not sure how to set this weight
                θ.logw[m] = 0.0
                acc[m] = 1
            end
        end
    end

    acc_rate = mean(acc) / M
    println(acc_rate)

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

    k = length(θ0)
    ξ = 1.e-12

    # initialize particles
    pθ = prior(θ0,Matrix{Float64}(I,k,k))
    θ = rand(pθ,M)
    θ = Particles([θ[:,m] for m in 1:M])

    # parallelize this
    for m in 1:M
        Θm = StateSpaceModel(model(θ.x[m]...))
        Xm = bootstrapFilter(N,y,Θm)
        θ.logw[m] = sum([Xmt.logμ for Xmt in Xm.p])
    end

    while ξ < 1.0
        # find exponent ξ, reweight, and resample
        newξ = gridSearch(ξ,θ,B)
        θ = reweight(θ,[(newξ-ξ)*θ.logw[m] for m in 1:M])

        if θ.ess < B*M
            println("resampling...")
            θ = resample(θ)
            θ = randomWalkMH(θ,model,N,y,pθ,newξ)
        end

        println(newξ)
        ξ = newξ
    end

    return θ
end