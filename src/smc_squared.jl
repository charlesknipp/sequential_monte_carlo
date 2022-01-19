# not sure this works, so I might need some testing
function randomWalk(θ::Particles,c::Float64=0.5)
    k = length(θ.x[1])
    x = reduce(hcat,θ.x)

    # calculate the weighted mean
    μ = [(sum(θ.w .* x[i,:]))/sum(θ.w) for i in 1:k]
    
    # calculate the weighted covariance
    adjx = [x[:,m] - μ for m in 1:length(θ.x)]
    Σ = sum(θ.w[m]*adjx[m]*adjx[m]' for m in 1:M)/sum(w)
    # Σ = StatsBase.cov(x,μ,θ.w)

    return rand(MvNormal(μ,c*Σ))
end

# not complete, and also not necessary YET
function bootstrapStep(
        t::Int64,
        Θ::Vector{StateSpaceModel},
        N::Int64,
        y::Vector{Float64},
        Xt::Vector{Particles},
        B::Float64 = 0.5
    )
    M = length(Xt)

    for m in 1:M
        xt = rand.(Θ[m].transition.(Xt[m].x),N)
        wt = logpdf.(Θ[m].observation.(xt),y[t])
        Xt[m] = resample(Particles(xt,wt),B)
    end

    return Xt
end

# I should write a method to perform a filter for t iterations
function SMC2(
        N::Int64,
        M::Int64,
        y::Vector{Float64},
        θ0::Vector{Float64},
        prior::Function,
        B::Float64 = 0.5,
        model = LinearGaussian
    )

    T = length(y)
    k = length(θ0)

    θ = prior(M,θ0,Matrix{Float64}(I,k,k))
    θ = Particles([θ[:,m] for m in 1:M])

    Θ = [StateSpaceModel(model(θ.p[m].x...)) for m in 1:M]

    x0 = (Θ[1].dim_x == 1) ? 0.0 : zeros(Float64,Θ[1].dim_x)
    Xt = [Particles(rand(Θ[m].transition(x0),N)) for m in 1:M]

    # perform iteration t of the bootstrap filter and reweight θ particles
    for t in 1:T
        Xt = bootstrapStep(t,Θ,N,y,Xt,B)
        θ  = reweight(θ,θ.logw+[Xtm.logμ for Xtm in Xt])
        
        # perform MH steps in case of degeneracy
        if θ.ess < B*N
            newθ = randomWalk(θ,0.5)
            newΘ = [StateSpaceModel(model(newθ.p[m].x...)) for m in 1:M]

            # perform another PF from 1:t
            newx0 = (newΘ[1].dim_x == 1) ? 0.0 : zeros(Float64,newΘ[1].dim_x)
            newXt = [Particles(rand(newΘ[m].transition(newx0),N)) for m in 1:M]

            for k in 1:t
                newXt = bootstrapStep(k,newΘ,N,y,newXt,B)
                newθ  = reweight(newθ,newθ.logw+[newXt[m].logμ for m in 1:M])
            end

            # Z(θ) ≡ likelihood
            Zt    = Xt.logμ
            newZt = newXt.logμ

            # p(θ) ≡ pdf of prior
            pt    = logpdf(prior,θ)
            newpt = logpdf(prior,newθ)

            # T(θ) ≡ pdf of proposal distribution
            Tt    = logpdf()
            newTt = logpdf()
        end
    end
end