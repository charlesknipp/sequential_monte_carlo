export StateSpaceModel,LinearGaussian,simulate,ModelParameters,AbstractSSM

abstract type AbstractSSM end
abstract type ModelParameters end

struct StateSpaceModel{F,G,F0} <: AbstractSSM
    # define general structure using functions
    transition::F
    observation::G
    initial_dist::F0
end

# add static array construction...this is pretty rough
struct LinearGaussian{T<:Real} <: ModelParameters
    A::Matrix{T}
    B::Matrix{T}

    Q::Matrix{T}
    R::Matrix{T}

    μ0::Array{T}
    Σ0::Matrix{T}
end

function StateSpaceModel(params::LinearGaussian)
    # import parameters
    A,B = params.A,params.B
    Q,R = params.Q,params.R

    f(xt) = MvNormal(A*xt,Q)
    g(xt) = MvNormal(B*xt,R)

    f0 = MvNormal(params.μ0,params.Σ0)

    return StateSpaceModel(f,g,f0)
end


function simulate(rng::AbstractRNG,mod::StateSpaceModel,T::Int)
    x = Vector{Vector{Float64}}(undef,T)
    y = similar(x)

    x[1] = rand(rng,mod.initial_dist)

    for t = 1:T-1
        y[t] = rand(rng,mod.observation(x[t]))
        x[t+1] = rand(rng,mod.transition(x[t]))
    end

    y[T] = rand(rng,mod.observation(x[T]))
    return x,y
end

dims(mod::LinearGaussian) = (size(mod.A,1),size(mod.B,1))
simulate(mod::StateSpaceModel,T::Int) = simulate(Random.GLOBAL_RNG,mod,T)