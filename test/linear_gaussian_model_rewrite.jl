using StaticArrays,Distributions,LinearAlgebra

const ScalarOrArray{T} = Union{T,Array{T}}

abstract type AbstractStateSpaceModel{T<:Real} end

struct LinearGaussian{T<:Real} <: AbstractStateSpaceModel{T}
    A::Matrix{T}
    B::Matrix{T}

    Q::Matrix{T}
    R::Matrix{T}

    μ0::Array{T}
    Σ0::Matrix{T}
end

#=
    Rewrite this in terms of static arrays to speed up the computation of the
    bootstrap filter methods.

    SMatrix and SVector are the object types I need to implement; there is
    potential to rewrite the method to avoid the use of reshaping to preserve
    the row/column structure of the measurement and transition densities.
=#
function LinearGaussian(
        A::ScalarOrArray{T},B::ScalarOrArray{T},
        Q::ScalarOrArray{T},R::ScalarOrArray{T},
        μ0::ScalarOrArray{T},Σ0::ScalarOrArray{T}
    ) where {T<:Real}

    # dimensions of recorded sequence
    dim_x = size(A,1)
    dim_y = size(B,1)

    # dimensions of random vectors
    dim_δ = size(Q,2)
    dim_ε = size(R,2)

    A = reshape(vcat(A),Size(dim_x,dim_x))
    B = reshape(vcat(B),Size(dim_y,dim_y))

    Q = reshape(vcat(Q),dim_x,dim_δ)
    R = reshape(vcat(R),dim_y,dim_ε)

    μ0 = reshape([μ0;],dim_x)
    Σ0 = reshape(vcat(Σ0),dim_x,dim_x)

    return LinearGaussian{T}(A,B,Q,R,μ0,Σ0)
end

# this method may not work, so I should play around with it
function LinearGaussian(
        A::ScalarOrArray{<:Real},B::ScalarOrArray{<:Real},
        Q::ScalarOrArray{<:Real},R::ScalarOrArray{<:Real},
        μ0::ScalarOrArray{<:Real},Σ0::ScalarOrArray{<:Real}
    )

    T = Base.promote_eltype(A,B,Q,R,μ0,Σ0)

    return LinearGaussian(
        convert(ScalarOrArray{T},A),convert(ScalarOrArray{T},B),
        convert(ScalarOrArray{T},Q),convert(ScalarOrArray{T},R),
        convert(ScalarOrArray{T},μ0),convert(ScalarOrArray{T},Σ0)
    )
end

dims(mod::LinearGaussian) = (size(mod.A,1),size(mod.B,1))


function transition(mod::LinearGaussian,xt::StaticVector)
    return MvNormal(mod.A*xt,mod.Q)
end

function observation(mod::LinearGaussian,xt::StaticVector)
    return MvNormal(mod.B*xt,mod.R)
end

function initialize(mod::LinearGaussian)
    return MvNormal(mod.μ0,mod.Σ0)
end


function simulate(rng::AbstractRNG,mod::AbstractStateSpaceModel,T::Int64)
    dx,dy = dims(mod)

    x = Matrix{Float64}(undef,dx,T)
    y = Matrix{Float64}(undef,dy,T)
    
    x[:,1] = rand(rng,initialize(mod))

    # try experimenting with views and inbounds
    for t in 1:T-1
        x[:,t+1] = rand(rng,transition(mod,x[:,t]))
    end

    for t in 1:T
        y[:,t] = rand(rng,observation(mod,x[:,t]))
    end

    return x,y
end


a = MvNormal(
    ones(SVector{2}),
    SMatrix{2,2}(1.0I(2))
)

@btime $a