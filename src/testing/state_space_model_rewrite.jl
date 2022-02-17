using Distributions,LinearAlgebra,BenchmarkTools

abstract type AbstractSSM end
abstract type ModelParameters end

struct StateSpaceModel{Model<:ModelParameters} <: AbstractSSM
    initial_dist::Sampleable

    # this is not the most efficient method, maybe use closures?
    transition::Function
    observation::Function
end

function StateSpaceModel{Model}(params::Vector) where Model
    return StateSpaceModel(Model(params...))
end

function reparameterize(model::StateSpaceModel,params::Vector)
    Model = typeof(model).parameters[1]
    return StateSpaceModel{Model}(params)
end

struct LinearGaussian <: ModelParameters
    # scalar multipliers
    A::Union{Float64,Matrix{Float64}}
    B::Union{Float64,Matrix{Float64}}

    # covariances
    Q::Union{Float64,Matrix{Float64}}
    R::Union{Float64,Matrix{Float64}}
end

function StateSpaceModel(params::LinearGaussian)
    dim_x = size(params.A,1)

    x0 = zeros(Float64,dim_x)
    Σ0 = Matrix{Float64}(I,dim_x,dim_x)

    dist = MvNormal(x0,Σ0)

    f(xt) = MvNormal(params.A*xt,params.Q)
    g(xt) = MvNormal(params.B*xt,params.R)

    return StateSpaceModel{LinearGaussian}(dist,f,g)
end

function simulate(model::StateSpaceModel,T::Int64)
    x0 = rand(model.initial_dist)
    y0 = rand(model.observation(x0))

    x = fill(x0,T)
    y = fill(y0,T)

    for t in 2:T
        x[t] = rand(model.transition(x[t-1]))
        y[t] = rand(model.observation(x[t]))
    end

    return (x,y)
end

function timeSSM()
    θ = [ones(Float64,2,2),ones(Float64,2,2),Matrix{Float64}(I(2)),Matrix{Float64}(I(2))]
    model_type = LinearGaussian(θ...)
    test_model = StateSpaceModel(model_type)
    return simulate(test_model,100)
end

@benchmark timeSSM()