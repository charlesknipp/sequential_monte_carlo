include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra,StatsBase

#=
function bootstrapStep(t::Int64,Θ::Vector{StateSpaceModel},y::Vector{Float64},Xt::ParticleSet)
    M = length(Xt.p)

    for m in 1:M
        xt = rand.(Θ[m].transition.(Xt.p[m].x))
        wt = logpdf.(Θ[m].observation.(xt),y[t])

        wt += Xt.p[m].logw
        Xt.p[m] = resample(Particles(xt,wt))
    end

    return Xt
end

function randomWalk(θ::Particles,c::Float64,kernel::Function)
    M = length(θ.x)
    x = reduce(hcat,θ.x)

    # calculate the weighted mean
    μ = vec(mean(x,weights(θ.w),2))
    
    # calculate the weighted covariance (please double check)
    Σ = cov(x,weights(θ.w),2)

    # finish this to generate a particle set
    pθ = kernel(μ,c*Σ)
    newθ = rand(pθ,M)
    return Particles([newθ[:,m] for m in 1:M]),pθ
end

function randomWalkMH(t::Int64,θ::Particles,Xt::ParticleSet,N::Int64,prior::Function,c::Float64=0.5,chain::Int64=10)
    # consider whether I should do this per particle or over the whole set
    k = length(θ)
    M = length(θ.x)

    for _ in 1:chain
        newθ,pθ = randomWalk(θ,c,prior)
        newΘ = [StateSpaceModel(LinearGaussian(newθ.x[m]...)) for m in 1:M]

        # perform another PF from 1:t and OPTIMIZE THIS
        newx0 = (newΘ[1].dim_x == 1) ? 0.0 : zeros(Float64,newΘ[1].dim_x)
        newXt = ParticleSet([Particles(rand(newΘ[m].transition(newx0),N)) for m in 1:M])

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

prior(μ,Σ) = TruncatedMvNormal(μ,Σ,[-1.0,-1.0,0.0,0.0],[1.0,1.0,Inf,Inf])

M = 100
N = 100
k = 4
θ0 = [0.7,0.7,1.0,1.0]
θ = rand(prior(θ0,0.6*Matrix{Float64}(I,k,k)),M)
θ = Particles([θ[:,m] for m in 1:M])


T = length(y)

Θ = [StateSpaceModel(LinearGaussian(θ.x[m]...)) for m in 1:M]
x0 = (Θ[1].dim_x == 1) ? 0.0 : zeros(Float64,Θ[1].dim_x)
Xt = ParticleSet([Particles(rand(Θ[m].transition(x0),N)) for m in 1:M])

for t in ProgressBar(1:T)
    global Xt = bootstrapStep(t,Θ,y,Xt)
    global θ  = reweight(θ,θ.logw+[Xtm.logμ for Xtm in Xt.p])
    
    # perform MH steps in case of degeneracy of θ particles
    if θ.ess < 0.5*M
        global θ,Xt = randomWalkMH(t,θ,Xt,N,prior,0.5,4)
    end
end
=#


# simulate shit
test_params = LinearGaussian(0.8,0.5,1.0,1.0)
test_model  = StateSpaceModel(test_params)
x,y = simulate(test_model,200)

prior(μ,Σ) = TruncatedMvNormal(μ,Σ,[-1.0,-1.0,0.0,0.0],[1.0,1.0,Inf,Inf])
θ0 = [0.7,0.7,1.0,1.0]

θ,Xt = SMC2(100,100,y,θ0,prior,0.5,LinearGaussian)

mean(reduce(hcat,θ.x),weights(θ.w),2)