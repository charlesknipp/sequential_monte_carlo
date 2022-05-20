# assuming variables are iid we can construct a product distribution which is
# more flexible and agnositc of reparameterizations

using StaticArrays
using Distributions
import PDMats
using Random
using LinearAlgebra

# Make distributions faster for static arrays

@inline Base.:(-)(x::StaticArray,_::Distributions.Zeros) = x
@inline Base.:(-)(::Distributions.Zeros,x::StaticArray) = -x
@inline Distributions.logpdf(d::Distribution,x,xp,t) = logpdf(d,x-xp)
@inline Distributions.sqmahal(d::MvNormal,x::StaticArray) = Distributions.invquad(d.Σ,x-d.μ)
@inline PDMats.invquad(a::PDMats.ScalMat,x::StaticVector) = dot(x,x) / a.value
@inline PDMats.invquad(a::PDMats.PDMat,x::StaticVector) = dot(x, a \ x)
@inline Base.:(\)(a::PDMats.PDMat,x::StaticVector) = a.chol \ x
@inline PDMats.invquad(a::PDMats.PDiagMat,x::StaticVector) = PDMats.wsumsq(1 ./ a.diag, x)



struct Mixed <: ValueSupport end

"""
    TupleProduct(v::NTuple{N,UnivariateDistribution})
Create a product distribution where the individual distributions are stored in a tuple. Supports mixed/hybrid Continuous and Discrete distributions
"""
struct TupleProduct{N,S,V<:NTuple{N,UnivariateDistribution}} <: MultivariateDistribution{S}
    v::V
    function TupleProduct(v::V) where {N,V<:NTuple{N,UnivariateDistribution}}
        all(Distributions.value_support(typeof(d)) == Discrete for d in v) &&
            return new{N,Discrete,V}(v)
        all(Distributions.value_support(typeof(d)) == Continuous for d in v) &&
            return new{N,Continuous,V}(v)
        return new{N,Mixed,V}(v)
    end
end

TupleProduct(d::Distribution...) = TupleProduct(d)
Base.length(d::TupleProduct{N}) where N = N

@generated function Distributions._rand!(rng::AbstractRNG, d::TupleProduct{N}, x::AbstractVector{<:Real}) where N
    quote
        Base.Cartesian.@nexprs $N i->(x[i] = rand(rng, d.v[i]))
        x
    end
end

@generated function Distributions._logpdf(d::TupleProduct{N}, x::AbstractVector{<:Real}) where N
    :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i], x[i]))
end

# To make it a bit faster also for the regular Product
@generated function Distributions._logpdf(d::Product, x::StaticVector{N}{<:Real}) where N
    :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i], x[i]))
end

Distributions.mean(d::TupleProduct) = vcat(mean.(d.v)...)
Distributions.var(d::TupleProduct) = vcat(var.(d.v)...)
Distributions.cov(d::TupleProduct) = Diagonal(var(d))
Distributions.entropy(d::TupleProduct) = sum(entropy, d.v)
Base.extrema(d::TupleProduct) = minimum.(d.v), maximum.(d.v)

@generated function Random.rand(rng::AbstractRNG, d::TupleProduct{N}) where N
    quote
        SVector(Base.Cartesian.@ntuple $N i->(rand(rng, d.v[i])))
    end
end

@btime MvNormal