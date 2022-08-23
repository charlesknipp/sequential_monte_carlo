export StateSpaceModel,LinearGaussian,StochasticVolatility
export simulate,transition,observation,initial_dist

abstract type ModelParameters end

struct StateSpaceModel{T<:ModelParameters}
    parameters::T
    dims::Tuple{Int64,Int64}
end

function simulate(rng::AbstractRNG,model::StateSpaceModel,T::Int64)
    x_dim,y_dim = model.dims

    x_type = x_dim == 1 ? Float64 : Vector{Float64}
    y_type = y_dim == 1 ? Float64 : Vector{Float64}

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
    return Normal(model.parameters.x0,model.parameters.Q)
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
unobserved component stochastic volatility

x[t] ~ N(x[t-1],exp(0.5*σx[t]))
y[t] ~ N(x[t],exp(0.5*σy[t]))

σx[t] ~ N(σx[t-1],γ)
σy[t] ~ N(σy[t-1],γ)
"""
struct UCSV <: ModelParameters
    # smoothing parameter
    γ::Float64

    # starting value
    x0::Float64

    # log volatilities
    σx::Base.RefValue{Float64}
    σy::Base.RefValue{Float64}
end

function transition(
        model::StateSpaceModel{UCSV},
        x::Float64
    )
    γ  = model.parameters.γ
    σx = model.parameters.σx
    σy = model.parameters.σy

    # update log volatilities
    σx[] += rand(Normal(0.0,γ))
    σy[] += rand(Normal(0.0,γ))

    return Normal(x,exp(0.5*σx[]))
end

function observation(
        model::StateSpaceModel{UCSV},
        x::Float64
    )
    σy = model.parameters.σy

    return Normal(x,exp(0.5*σy[]))
end

function initial_dist(
        model::StateSpaceModel{UCSV}
    )
    γ  = model.parameters.γ
    σx = model.parameters.σx
    σy = model.parameters.σy
    x0 = model.parameters.x0

    # update log volatilities
    σx[] += rand(Normal(0.0,γ))
    σy[] += rand(Normal(0.0,γ))

    return Normal(x0,exp(0.5*σx[]))
end
