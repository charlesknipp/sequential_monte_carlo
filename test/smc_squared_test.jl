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

    Θ = [StateSpaceModel(model(θ.x[m]...)) for m in 1:M]

    x0 = (Θ[1].dim_x == 1) ? 0.0 : zeros(Float64,Θ[1].dim_x)
    Xt = [Particles(rand(Θ[m].transition(x0),N)) for m in 1:M]

    # perform iteration t of the bootstrap filter and reweight θ particles
    for t in 1:T
        Xt = bootstrapStep(t,Θ,N,y,Xt,B)
        θ  = reweight(θ,θ.logw+[Xtm.logμ for Xtm in Xt])
        
        # perform MH steps in case of degeneracy of θ particles
        if θ.ess < B*M
            θ = randomWalkMH(t,θ,Xt,N,0.5,10)
        end
    end
end

include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions,LinearAlgebra

prior(μ,Σ) = TruncatedMvNormal(μ,Σ,[-1.0,-1.0,0.0,0.0],[1.0,1.0,Inf,Inf])

M = 100
N = 100
k = 4
θ0 = [0.7,0.7,1.0,1.0]
θ = rand(prior(θ0,Matrix{Float64}(I,k,k)),M)
θ = Particles([θ[:,m] for m in 1:M])

# simulate shit
test_params = LinearGaussian(0.8,0.5,1.0,1.0)
test_model  = StateSpaceModel(test_params)
x,y = simulate(test_model,100)
T = length(y)

Θ = [StateSpaceModel(LinearGaussian(θ.x[m]...)) for m in 1:M]
x0 = (Θ[1].dim_x == 1) ? 0.0 : zeros(Float64,Θ[1].dim_x)
Xt = [Particles(rand(Θ[m].transition(x0),N)) for m in 1:M]

for t in 1:T
    Xt = bootstrapStep(t,Θ,N,y,Xt,B)
    θ  = reweight(θ,θ.logw+[Xtm.logμ for Xtm in Xt])
    
    # perform MH steps in case of degeneracy of θ particles
    if θ.ess < B*M
        θ = randomWalkMH(t,θ,Xt,N,0.5,10)
    end
end