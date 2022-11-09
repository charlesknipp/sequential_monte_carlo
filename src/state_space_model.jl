export StateSpaceModel,LinearGaussian,StochasticVolatility,StochasticVolatilityLeverage,UC,UCSV
export simulate,transition,observation,initial_dist,preallocate

abstract type ModelParameters end

struct StateSpaceModel{T<:ModelParameters}
    parameters::T
    dims::Tuple{Int64,Int64}
end

function simulate(rng::AbstractRNG,model::StateSpaceModel,T::Int64)
    x_dim,y_dim = model.dims

    x_type = x_dim == 1 ? Float64 : SVector{x_dim,Float64}
    y_type = y_dim == 1 ? Float64 : SVector{y_dim,Float64}

    x = Vector{x_type}(undef,T)
    y = Vector{y_type}(undef,T)

    x[1] = rand(rng,initial_dist(model))
    y[1] = rand(rng,observation(model,x[1]))

    for t in 2:T
        x[t] = rand(rng,transition(model,x[t-1]))
        y[t] = rand(rng,observation(model,x[t]))
    end

    return x,y
end

simulate(mod::StateSpaceModel,T::Int64) = simulate(Random.GLOBAL_RNG,mod,T)


"""
univariate linear gaussian

x[t] ~ N(A*x[t-1],Q)
y[t] ~ N(B*x[t],R)
"""
struct LinearGaussian <: ModelParameters
    # coefficients
    A::Float64
    B::Float64

    # variances
    Q::Float64
    R::Float64

    # initial distribution
    x0::Float64
    σ0::Float64
end

function LinearGaussian(;A,B,Q,R,x0=0.0,σ0=1.0)
    return LinearGaussian(A,B,Q,R,x0,σ0)
end

function preallocate(model::StateSpaceModel{LinearGaussian},N::Int64)
    return zeros(Float64,N)
end

function transition(
        model::StateSpaceModel{LinearGaussian},
        x::Float64
    )
    A = model.parameters.A
    Q = model.parameters.Q

    return Normal(A*x,Q)
end

function observation(
        model::StateSpaceModel{LinearGaussian},
        x::Float64
    )
    B = model.parameters.B
    R = model.parameters.R

    return Normal(B*x,R)
end

function initial_dist(model::StateSpaceModel{LinearGaussian})
    return Normal(model.parameters.x0,model.parameters.σ0)
end

"""
stochastic volatility

x[t] ~ N(μ+ρ*(μ-x[t-1]),σ)
y[t] ~ N(0,exp(0.5*x[t]))
"""
struct StochasticVolatility <: ModelParameters
    # unconditional mean and speed
    μ::Float64
    ρ::Float64

    # volatility
    σ::Float64
end

function StochasticVolatility(;μ,ρ,σ)
    return StochasticVolatility(μ,ρ,σ)
end

function preallocate(model::StateSpaceModel{StochasticVolatility},N::Int64)
    return zeros(Float64,N)
end

function transition(
        model::StateSpaceModel{StochasticVolatility},
        x::Float64
    )
    μ = model.parameters.μ
    ρ = model.parameters.ρ
    σ = model.parameters.σ

    return Normal(μ+ρ*(x-μ),σ)
end

function observation(
        model::StateSpaceModel{StochasticVolatility},
        x::Float64
    )
    return Normal(0.0,exp(0.5*x))
end

function initial_dist(model::StateSpaceModel{StochasticVolatility})
    μ = model.parameters.μ
    ρ = model.parameters.ρ
    σ = model.parameters.σ

    return Normal(μ,σ/sqrt(1.0-ρ^2))
end

"""
stochastic volatility with leverage

x[t] = μ+ρ*(μ-x[t-1])+σ*u[t]
y[t] = exp(0.5*x[t])*v[t]

cor(u[t],v[t]) = ϕ
"""
struct StochasticVolatilityLeverage <: ModelParameters
    # unconditional mean and speed
    μ::Float64
    ρ::Float64

    # volatility
    σ::Float64

    # leverage
    ϕ::Float64
end

function StochasticVolatilityLeverage(;μ,ρ,σ,ϕ)
    return StochasticVolatilityLeverage(μ,ρ,σ,ϕ)
end

function preallocate(model::StateSpaceModel{StochasticVolatilityLeverage},N::Int64)
    return fill(zeros(SVector{2}),N)
end

function transition(
        model::StateSpaceModel{StochasticVolatilityLeverage},
        x::SVector{2,Float64}
    )
    μ = model.parameters.μ
    ρ = model.parameters.ρ
    σ = model.parameters.σ

    return TupleProduct((
        Dirac(μ+ρ*(x[2]-μ)),
        Normal(μ+ρ*(x[2]-μ),σ)
    ))
end

function observation(
        model::StateSpaceModel{StochasticVolatilityLeverage},
        x::SVector{2,Float64}
    )
    ϕ = model.parameters.ϕ
    σ = model.parameters.σ

    u  = (x[2]-x[1]) / σ
    σx = exp(0.5*x[2])
    
    return Normal(u*ϕ*σx,σx*(1-ϕ^2))
end

function initial_dist(model::StateSpaceModel{StochasticVolatilityLeverage})
    μ = model.parameters.μ
    ρ = model.parameters.ρ
    σ = model.parameters.σ

    return return TupleProduct((
        Dirac(μ),
        Normal(μ,σ/sqrt(1.0-ρ^2))
    ))
end

"""
basic unobserved component model

x[t] ~ N(x[t-1],σx)
y[t] ~ N(x[t],σy)

cov(σx,σy) == 0
"""
struct UC <: ModelParameters
    # initial trend value
    x0::Float64

    # uncorrelated disturbances
    σy::Float64
    σx::Float64
end

function preallocate(model::StateSpaceModel{UC},N::Int64)
    return zeros(Float64,N)
end

function transition(
        model::StateSpaceModel{UC},
        x::Float64
    )
    σx = model.parameters.σx

    return Normal(x,σx)
end

function observation(
        model::StateSpaceModel{UC},
        x::Float64
    )
    σy = model.parameters.σy

    return Normal(x,σy)
end

function initial_dist(
        model::StateSpaceModel{UC}
    )
    σx = model.parameters.σx
    x0 = model.parameters.x0
    return Normal(x0,σx)
end

"""
unobserved component stochastic volatility

x[t] ~ N(x[t-1],exp(0.5*σx[t-1]))
y[t] ~ N(x[t],exp(0.5*σy[t]))

σx[t] ~ N(σx[t-1],γ)
σy[t] ~ N(σy[t-1],γ)
"""
struct UCSV <: ModelParameters
    # smoothing parameter
    γ::Float64

    # starting value
    x0::Float64
    σ0::Tuple{Float64,Float64}
end
        
function preallocate(model::StateSpaceModel{UCSV},N::Int64)
    return fill(zeros(SVector{3}),N)
end

function transition(
        model::StateSpaceModel{UCSV},
        x::SVector{3,Float64}
    )
    γ = model.parameters.γ
    x,σx,σy = x

    return TupleProduct((
        Normal(x,exp(0.5*σx)),
        Normal(σx,γ),
        Normal(σy,γ)
    ))
end

function observation(
        model::StateSpaceModel{UCSV},
        x::SVector{3,Float64}
    )
    x,_,σy = x

    return Normal(x,exp(0.5*σy))
end

function initial_dist(
        model::StateSpaceModel{UCSV}
    )
    γ  = model.parameters.γ
    x0 = model.parameters.x0
    σx,σy = model.parameters.σ0

    return TupleProduct((
        Normal(x0,exp(0.5*σx)),
        Normal(σx,γ),
        Normal(σy,γ)
    ))
end
