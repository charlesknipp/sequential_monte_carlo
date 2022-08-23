export TupleProduct

"""
Mixed value support indicates that the distribution is a mix of continuous and discrete dimensions.
"""
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
# Distributions._rand!(rng::AbstractRNG, d::TupleProduct, x::AbstractVector{<:Real}) =     broadcast!(dn->rand(rng, dn), x, d.v)

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
        SVector(Base.Cartesian.@ntuple $N i->(rand(rng,d.v[i])))
    end
end