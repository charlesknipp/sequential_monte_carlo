export TruncatedMvNormal

struct TruncatedMvNormal{T<:Real,Cov<:AbstractMatrix{T},Mean<:AbstractVector{T}} <: ContinuousMultivariateDistribution
    # distribution parameters
    μ::Mean
    Σ::Cov

    # lower and upper bounds of the distribution
    l::Vector{T}
    u::Vector{T}
end

# define some default constructors
function TruncatedMvNormal(
        μ::AbstractVector{T},
        Σ::AbstractMatrix{T},
        l::AbstractVector{T},
        u::AbstractVector{T}
    ) where {T<:Real}

    dim(Σ) == length(μ) || throw(DimensionMismatch("mu and Sigma dimensions mismatch"))
    length(l) == length(u) || throw(DimensionMismatch("limit dimensions mismatch"))
    length(μ) == length(u) || throw(DimensionMismatch("limit and parameter dimensions mismatch"))

    return TruncatedMvNormal{T,typeof(Σ),typeof(μ)}(μ,Σ,l,u)
end

Base.eltype(::Type{<:TruncatedMvNormal{T}}) where {T} = T

function Base.show(io::IO,d::TruncatedMvNormal)
    params = [
        (:dim,length(d)),(:μ,mean(d)),(:Σ,cov(d)),
        (:lower,d.l),(:upper,d.u)
    ]

    return Distributions.show_multline(io,d,params)
end

Base.length(d::TruncatedMvNormal) = length(d.μ)
mean(d::TruncatedMvNormal) = d.μ
params(d::TruncatedMvNormal) = (d.μ,d.Σ)

var(d::TruncatedMvNormal) = diag(d.Σ)
cov(d::TruncatedMvNormal) = Matrix(d.Σ)
_cov(d::TruncatedMvNormal) = d.Σ

invcov(d::TruncatedMvNormal) = Matrix(inv(d.Σ))
logdetcov(d::TruncatedMvNormal) = logdet(d.Σ)

function Distributions._logpdf(d::TruncatedMvNormal,y::AbstractVector)
    k = length(d)
    x = d.μ

    logprob = 0

    for i in 1:k
        # convert covariance into a block matrix
        Σ11,Σ22 = d.Σ[i,i],d.Σ[1:end .!= i,1:end .!= i]
        Σ12,Σ21 = d.Σ[1:end .!= i,i]',d.Σ[1:end .!= i,i]

        x2 = x[1:end .!= i]
        μ1,μ2 = d.μ[i],d.μ[1:end .!= i]

        condΣ = Σ11 - Σ12*inv(Σ22)*Σ21
        condμ = μ1 + Σ12*inv(Σ22)*(x2-μ2)

        p1 = logpdf(Normal(condμ,sqrt(condΣ)),y[i])
        p2 = log(cdf(Normal(),d.u[i])-cdf(Normal(),d.l[i]))
        logprob += (p1-p2)

        x[i] = y[i]
    end

    return logprob
end

# extend to include some AbstractRNG input
function Base.rand(d::TruncatedMvNormal,n::Int64=1)
    k = length(d)
    x = ones(Float64,k,n).*(d.μ)

    # perform this N times over; either vectorize or iterate over another loop
    for i in 1:k
        # convert covariance into a block matrix
        Σ11,Σ22 = d.Σ[i,i],d.Σ[1:end .!= i,1:end .!= i]
        Σ12,Σ21 = d.Σ[1:end .!= i,i]',d.Σ[1:end .!= i,i]

        x2 = x[1:end .!= i,1]
        μ1,μ2 = d.μ[i],d.μ[1:end .!= i]

        condΣ = Σ11 - Σ12*inv(Σ22)*Σ21
        condμ = μ1 + Σ12*inv(Σ22)*(x2-μ2)

        lcdf = cdf(Normal(condμ,sqrt(condΣ)),d.l[i])
        ucdf = cdf(Normal(condμ,sqrt(condΣ)),d.u[i])
        prob = Base.rand(Uniform(lcdf,ucdf),n)

        x[i,:] = quantile(Normal(),prob)*sqrt(condΣ) .+ condμ
    end

    return x
end