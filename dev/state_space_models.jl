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

    for t in 1:T-1
        y[t]   = rand(rng,observation(model,x[t]))
        x[t+1] = rand(rng,transition(model,x[t]))
    end

    y[T] = rand(rng,observation(model,x[T]))

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
    σ0::Float64
end

function transition(
        model::StateSpaceModel{LinearGaussian},
        xt::Float64
    )

    A = model.parameters.A
    Q = model.parameters.Q

    return Normal(A*xt,Q)
end

function observation(
        model::StateSpaceModel{LinearGaussian},
        xt::Float64
    )

    B = model.parameters.B
    R = model.parameters.R

    return Normal(B*xt,R)
end

function initial_dist(model::StateSpaceModel{LinearGaussian})
    return Normal(model.parameters.x0,model.parameters.σ0)
end