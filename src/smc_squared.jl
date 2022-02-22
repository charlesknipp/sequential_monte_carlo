export SMC2

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

# FIX: this is the most inefficient step in the entire process
function randomWalkMH(
        t::Int64,
        θ::Particles,
        Xt::ParticleSet,
        y::Vector{Float64},
        N::Int64,
        prior::Function,
        model,
        B::Float64,
        c::Float64=0.5,
        chain::Int64=10
    )

    # consider whether I should do this per particle or over the whole set
    M = length(θ.x)

    for _ in 1:chain
        newθ,pθ = randomWalk(θ,prior,c)
        newΘ = [StateSpaceModel(model(newθ.x[m]...)) for m in 1:M]

        # perform another PF from 1:t and OPTIMIZE THIS
        x0 = (newΘ[1].dim_x == 1) ? 0.0 : zeros(Float64,newΘ[1].dim_x)
        newXt = [Particles(rand(newΘ[m].transition(x0),N)) for m in 1:M]
        newXt = ParticleSet(newXt)

        for k in 1:t
            newXt = bootstrapStep(k,newΘ,y,newXt,B)
            newwt = newθ.logw + [newXt.p[m].logμ for m in 1:M]
            newθ  = Particles(newθ.x,newwt)
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
                particles[m]   = newθ.x[m]
                log_weights[m] = 0.0

                Xt.p[m] = newXt.p[m]
            end
        end

        θ = Particles(particles,log_weights)
    end

    return θ,Xt
end

# this works surprisingly well
function bootstrapStep(
        t::Int64,
        Θ::Vector{StateSpaceModel},
        y::Vector{Float64},
        Xt::ParticleSet,
        B::Float64
    )
    M = length(Xt)

    # could parallelize
    for m in 1:M
        # resample the previous state and propogate forward
        xt = resample(Xt.p[m],B)
        xt = rand.(Θ[m].transition.(xt.x))

        # weight based on the likelihood of observation
        wt = logpdf.(Θ[m].observation.(xt),y[t])

        # create a particle system with appropriate weights
        Xt.p[m] = Particles(xt,wt)
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
        # resample with a random walk MH kernel to avoid degeneracy
        if θ.ess < B*M
            θ,Xt = randomWalkMH(t,θ,Xt,y,N,prior,model,0.5,0.8,3)
        end

        # propogate forward and reweight
        Xt = bootstrapStep(t,Θ,y,Xt,B)
        wt = θ.logw + [Xt.p[m].logμ for m in 1:M]
        θ  = Particles(θ.x,wt)
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
    θ0 = Particles([θ0[:,m] for m in 1:M])

    return SMC2(N,M,y,θ0,prior,B,model)
end

#=
IDEA 1: consider changing Base.iterate for particles such that the filter will
    run at each iteration, and no waste is stored in between iterations

IDEA 2: create an object type for SMC² that stores exogenous characteristics of
    the algorithm (number of θ particles, number of X particles, model, etc)
=#