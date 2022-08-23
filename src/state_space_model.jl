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
π[t] = τ[t] + η[t]      s.t. η[t] ~ N(0,σ²[η,t]*ζ[η,t])
τ[t] = τ[t-1] + ε[t]    s.t. ε[t] ~ N(0,σ²[ε,t]*ζ[ε,t])

log(σ²[η,t]) = log(σ²[η,t-1]) + v[η,t]
log(σ²[ε,t]) = log(σ²[ε,t-1]) + v[ε,t]
"""
struct UCSV <: ModelParameters
    # smoothness parameter
    γ::Float64
end

function transition(
        model::StateSpaceModel{UCSV},
        x::Vector{Float64}
    )
    γ = model.parameters.γ
    x,ση,σε = x

    ση = ση + γ*rand()
end

function observation(
        model::StateSpaceModel{UCSV},
        x::Float64
    )
    # π[t+1] = π[t] + exp(x[1,t])
    # y[t] = π[t] + exp(x[2,t])
end