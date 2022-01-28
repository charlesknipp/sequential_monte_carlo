export SMC2

# this only works with particles that are Vector{Float64}
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


function randomWalkMH(
        t::Int64,
        θ::Particles,
        Xt::ParticleSet,
        N::Int64,
        prior::Function,
        c::Float64=0.5,
        chain::Int64=10
    )

    # consider whether I should do this per particle or over the whole set
    k = length(θ.x[1])
    M = length(θ.x)

    for _ in 1:chain
        newθ,pθ = randomWalk(θ,c,prior)
        newΘ = [StateSpaceModel(model(newθ.x[m]...)) for m in 1:M]

        # perform another PF from 1:t and OPTIMIZE THIS
        newx0 = (newΘ[1].dim_x == 1) ? 0.0 : zeros(Float64,newΘ[1].dim_x)
        newXt = [Particles(rand(newΘ[m].transition(newx0),N)) for m in 1:M]

        for k in 1:t
            newXt = bootstrapStep(k,newΘ,y,newXt)
            newθ  = reweight(newθ,newθ.logw+[newXt.p[m].logμ for m in 1:M])
        end

        # Z(θ) ≡ likelihood, p(θ) ≡ pdf of prior
        logZt = [Xt.p[m].logμ-newXt.p[m].logμ for m in 1:M]
        logpt = [logpdf(pθ,θ.x[m])-logpdf(pθ,newθ.x[m]) for m in 1:M]

        # acceptance ratio
        α = exp.(logZt+logpt)
        u = rand(M)

        particles   = θ.x
        log_weights = θ.logw

        for m in 1:M
            if u[m] ≤ minimum([α[m],1.0])
                # resample particle and set the weight w[m] ↤ 1
                particles[m]   = newθ[m]
                log_weights[m] = 0.0

                Xt[m] = newXt[m]
            end
        end

        θ = Particles(particles,log_weights)
    end

    return θ,Xt
end

function bootstrapStep(
        t::Int64,
        Θ::Vector{StateSpaceModel},
        y::Vector{Float64},
        Xt::ParticleSet
    )
    M = length(Xt)

    for m in 1:M
        xt = rand.(Θ[m].transition.(Xt.p[m].x))
        wt = logpdf.(Θ[m].observation.(xt),y[t])

        wt += Xt.p[m].logw
        Xt.p[m] = resample(Particles(xt,wt))
    end

    return Xt
end

# Base function for SMC² which takes a Particles object as input
function SMC2(
        N::Int64,
        M::Int64,
        y::Vector{Float64},
        θ::Particles,
        prior::Function,
        B::Float64 = 0.5,
        model = LinearGaussian
    )
    T = length(y)
    
    Θ  = [StateSpaceModel(model(θ.x[m]...)) for m in 1:M]
    x0 = (Θ[1].dim_x == 1) ? 0.0 : zeros(Float64,Θ[1].dim_x)
    Xt = ParticleSet([Particles(rand(Θ[m].transition(x0),N)) for m in 1:M])

    # perform iteration t of the bootstrap filter and reweight θ particles
    for t in ProgressBar(1:T)
        # perform MH steps in case of degeneracy of θ particles
        if θ.ess < B*M
            θ,Xt = randomWalkMH(t,θ,Xt,N,prior,0.5,5)
        end

        Xt = bootstrapStep(t,Θ,y,Xt)
        θ  = reweight(θ,θ.logw+[Xtm.logμ for Xtm in Xt.p])
    end

    return θ,Xt
end

# this is a wrapper given a guess for the initial state θ_0
function SMC2(
        N::Int64,
        M::Int64,
        y::Vector{Float64},
        θ0::Vector{Float64},
        prior::Function,
        B::Float64 = 0.5,
        model = LinearGaussian
    )
    k = length(θ0)

    θ0 = rand(prior(θ0,Matrix{Float64}(I,k,k)),M)
    θ0 = Particles([θ[:,m] for m in 1:M])

    return SMC2(N,M,y,θ0,prior,B,model)
end