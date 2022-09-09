export TupleProduct
using PDMats

# extend base operations for Distributions with static arrays
@inline Base.:(-)(x::StaticArray,::Distributions.Zeros) = x
@inline Base.:(-)(::Distributions.Zeros,x::StaticArray) = -x
@inline Base.:(\)(a::PDMats.PDMat,x::StaticVector) = a.chol \ x

# define square mahalanobis distance with static arrays
@inline function Distributions.sqmahal(
        d::MvNormal,
        x::StaticArray
    )
    return Distributions.invquad(d.Σ,x-d.μ)
end

# define inverse quadratic form with static vectors and scaling matrices
@inline function PDMats.invquad(
        a::PDMats.ScalMat,
        x::StaticVector
    )
    return dot(x,x) / a.value
end

# define inverse quadratic form with static vectors and positive definite
# matrices
@inline function PDMats.invquad(
        a::PDMats.PDMat,
        x::StaticVector
    )
    return dot(x,a\x)
end

# define inverse quadratic form with static vectors and positive definite
# diagonal matrices
@inline function PDMats.invquad(
        a::PDMats.PDiagMat,
        x::StaticVector
    )
    return PDMats.wsumsq(1 ./ a.diag, x)
end


struct Mixed <: ValueSupport end

struct TupleProduct{N,S,V<:NTuple{N,UnivariateDistribution}} <: MultivariateDistribution{S}
    v::V

    # define a default constructor conditional on cumulative dist types V
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

# define basic random generation for this type
@generated function Distributions._rand!(
        rng::AbstractRNG,
        d::TupleProduct{N},
        x::AbstractVector{<:Real}
    ) where N
    return quote
        Base.Cartesian.@nexprs $N i->(x[i] = rand(rng,d.v[i]))
        x
    end
end

# define random generation to work with static vectors
@generated function Random.rand(
        rng::AbstractRNG,
        d::TupleProduct{N}
    ) where N
    return quote
        SVector(Base.Cartesian.@ntuple $N i->(rand(rng,d.v[i])))
    end
end

@generated function Distributions._logpdf(
        d::TupleProduct{N},
        x::AbstractVector{<:Real}
    ) where N
    return :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i],x[i]))
end

# extend static array support to regular product distributions as well
@generated function Distributions._logpdf(
        d::Product,
        x::StaticVector{N}{<:Real}
    ) where N
    return :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i],x[i]))
end

# define summary statistics for tuple products
Distributions.mean(d::TupleProduct) = vcat(mean.(d.v)...)
Distributions.var(d::TupleProduct)  = vcat(var.(d.v)...)
Distributions.cov(d::TupleProduct)  = Diagonal(var(d))

# define the entropy
Distributions.entropy(d::TupleProduct) = sum(entropy, d.v)

# further extend base methods
Base.extrema(d::TupleProduct) = minimum.(d.v), maximum.(d.v)