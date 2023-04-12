export simulate,preallocate,transition,observation,initial_dist
export StateSpaceModel,LinearModel,UnivariateLinearGaussian,MultivariateLinearGaussian
export unobserved_components,hodrick_prescott,unobserved_components_stochastic_volatility

using Distributions
using LinearAlgebra
using Random

abstract type StateSpaceModel end

function simulate(rng::AbstractRNG,model::StateSpaceModel,T::Int64)
    XT,YT = get_types(model)

    x = Vector{XT}(undef,T)
    y = Vector{YT}(undef,T)

    x[1] = rand(rng,initial_dist(model))
    y[1] = rand(rng,observation(model,x[1]))

    for t in 2:T
        x[t] = rand(rng,transition(model,x[t-1]))
        y[t] = rand(rng,observation(model,x[t]))
    end

    return x,y
end

simulate(mod::StateSpaceModel,T::Int64) = simulate(Random.GLOBAL_RNG,mod,T)

function preallocate(model::StateSpaceModel,N::Int64)
    # not implemented yet
end

function transition(model::StateSpaceModel,x)
    # not implemented yet
end
function observation(model::StateSpaceModel,x)
    # not implemented yet
end
function initial_dist(model::StateSpaceModel,x)
    # not implemented yet
end

## LINEAR MODELS ##############################################################

struct LinearModel{AT,BT,QT,RT,XT,ΣT} <: StateSpaceModel
    # coefficients
    A::AT
    B::BT

    # variances
    Q::QT
    R::RT

    # initial distribution
    x0::XT
    σ0::ΣT
end

# we only expect the observation to be univariate
function get_types(
        model::LinearModel{AT,BT,QT,RT,XT,ΣT}
    ) where {AT,BT,QT,RT,XT,ΣT}
    return (XT,Float64)
end


"""
univariate linear gaussian

x[t] ~ N(A*x[t-1],Q)
y[t] ~ N(B*x[t],R)
"""
function UnivariateLinearGaussian(;A,B,Q,R,x0=0.0,σ0=1.0)
    T = Float64
    return LinearModel{T,T,T,T,T,T}(A,B,Q,R,x0,σ0)
end

# for univariate types let's define the following operations for linear models
function preallocate(
        model::LinearModel{AT,BT,QT,RT,Float64,Float64},
        N::Int64
    ) where {AT,BT,QT,RT}
    return zeros(Float64,N)
end

function transition(
        model::LinearModel{AT,BT,QT,RT,Float64,Float64},
        x::Float64
    ) where {AT,BT,QT,RT}
    A = model.A
    Q = model.Q
    return Normal(A*x,sqrt(Q))
end

function observation(
        model::LinearModel{AT,BT,QT,RT,Float64,Float64},
        x::Float64
    ) where {AT,BT,QT,RT}
    B = model.B
    R = model.R
    return Normal(B*x,sqrt(R))
end

function initial_dist(
        model::LinearModel{AT,BT,QT,RT,Float64,Float64}
    ) where {AT,BT,QT,RT}
    return Normal(model.x0,sqrt(model.σ0))
end

"""
local-level unobserved components model

x[t] ~ N(x[t-1],σx)
y[t] ~ N(x[t],σy)

cov(σx,σy) == 0
"""
function unobserved_components(;σε::T,ση::T,x0::T) where T
    return UnivariateLinearGaussian(
        A = one(T),
        B = one(T),
        Q = σε,
        R = ση,
        x0 = x0,
        σ0 = σε
    )
end


"""
multivariate linear gaussian

x[t] ~ N(A*x[t-1],Q)
y[t] ~ N(B*x[t],R)
"""
function MultivariateLinearGaussian(;A,B,Q,R,X0=zeros(size(A,1)),Σ0=1.0I(size(A,1)))
    # get coefficients
    AT = typeof(A)
    BT = typeof(B)

    # get variances
    QT = typeof(Q)
    RT = typeof(R)

    # get initial types
    XT = typeof(X0)
    ΣT = typeof(Σ0)

    # raise error if...

    return LinearModel{AT,BT,QT,RT,XT,ΣT}(A,B,Q,R,X0,Σ0)
end

# for multivariate types let's define the following operations for linear models
function preallocate(
        model::LinearModel{AT,BT,QT,RT,XT,ΣT},
        N::Int64
    ) where {AT,BT,QT,RT,XT<:AbstractArray,ΣT<:AbstractArray}
    return fill(zero(model.x0),N)
end

function transition(
        model::LinearModel{AT,BT,QT,RT,XT,ΣT},
        x::XT
    ) where {AT,BT,QT,RT,XT<:AbstractArray,ΣT<:AbstractArray}
    A = model.A
    Q = model.Q
    return MvNormal(A*x,Q)
end

function observation(
        model::LinearModel{AT,BT,QT,RT,XT,ΣT},
        x::XT
    ) where {AT,BT,QT,RT,XT<:AbstractArray,ΣT<:AbstractArray}
    B = model.B
    R = model.R
    return Normal(B*x...,R...)
end

function initial_dist(
        model::LinearModel{AT,BT,QT,RT,XT,ΣT}
    ) where {AT,BT,QT,RT,XT<:AbstractArray,ΣT<:AbstractArray}
    return MvNormal(model.x0,model.σ0)
end

"""
hodrick-prescott model

x[t] ~ N(2*x[t-1]-x[t-2],1/λ)
y[t] ~ N(x[t],1)
"""
function hodrick_prescott(;λ::ΛT,y::Vector{YT},init_cov=1000.0) where {ΛT<:Real,YT<:Real}
    return MultivariateLinearGaussian(
        A = [2.0 -1.0;1.0 0.0],
        B = [1.0 0.0],
        Q = [1/λ 0.0;0.0 0.0],
        R = [1.0],
        X0 = [3*y[1]-2*y[2],2*y[1]-y[2]],
        Σ0 = Matrix(init_cov*I(2))
    )
end

## NON-LINEAR MODELS ##########################################################

"""
unobserved components stochastic volatility

x[t] ~ N(x[t-1],(σε[t-1])²)
y[t] ~ N(x[t],(ση[t])²)

log(σε[t]) ~ N(log(σε[t-1]),γε)
log(ση[t]) ~ N(log(ση[t-1]),γη)
"""
struct UCSV <: StateSpaceModel
    # log volatility transition noise
    γ::Tuple{Float64,Float64}

    # initial values
    x0::Float64
    log_σ0::Tuple{Float64,Float64}
end

# function wrapper for use in model definition
function unobserved_components_stochastic_volatility(;x0,γε,γη,log_σε,log_ση)
    return UCSV((γε,γη),x0,(log_σε,log_ση))
end

function preallocate(model::UCSV,N::Int64)
    return fill(zeros(Float64,3),N)
end

function transition(model::UCSV,x::Vector{Float64})
    x,log_σε,log_ση = x
    γε,γη   = model.γ

    return TupleProduct((
        Normal(x,exp(0.5*log_σε)),
        Normal(log_σε,γε),
        Normal(log_ση,γη)
    ))
end

function observation(model::UCSV,x::Vector{Float64})
    x,log_σε,log_ση = x
    return Normal(x,exp(0.5*log_ση))
end

function initial_dist(model::UCSV)
    x0 = model.x0
    log_σε,log_ση = model.log_σ0
    γε,γη = model.γ

    return TupleProduct((
        Normal(x0,exp(0.5*log_σε)),
        Normal(log_σε,γε),
        Normal(log_ση,γη)
    ))
end

function get_types(model::UCSV)
    return (Vector{Float64},Float64)
end
