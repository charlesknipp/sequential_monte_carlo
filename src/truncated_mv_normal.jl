using Distributions

function mvTruncated(d::MultivariateDistribution,l::Vector{Real},u::Vector{Real})
    return truncated(d,promote(l,u)...)
end

function mvTruncated(d::MultivariateDistribution,l::Vector{T},u::Vector{T}) where {T <: Real}
    for i in 1:length(l)
        l[i] <= u[i] || error("the lower bound must be less or equal than the upper bound")
    end

    # (log)lcdf = (log) P(X < l) where X ~ d
    loglcdf = if value_support(typeof(d)) === Discrete
        logsubexp(logcdf(d,l),logpdf(d,l))
    else
        logcdf(d,l)
    end
    lcdf = exp(loglcdf)

    # (log)ucdf = (log) P(X ? u) where X ~ d
    logucdf = logcdf(d,u)
    ucdf = exp(logucdf)

    # (log)tp = (log) P(l ? X ? u) where X ? d
    logtp = logsubexp(loglcdf,logucdf)
    tp = exp(logtp)

    Truncated(d,promote(l,u,lcdf,ucdf,tp,logtp)...)
end

mvTruncated(d::MultivariateDistribution,l::Vector{Integer},u::Vector{Integer}) = truncated(d,float.(l),float.(u))

struct MvTruncated{D<:MultivariateDistribution,S<:ValueSupport,T<:Real} <: MultivariateDistribution{S}
    untruncated::D      # the original distribution (untruncated)
    lower::Vector{T}      # lower bound
    upper::Vector{T}      # upper bound
    lcdf::T       # cdf of lower bound
    ucdf::T       # cdf of upper bound

    tp::T         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::T      # log(tp), i.e. log(ucdf - lcdf)

    function Truncated(d::UnivariateDistribution,l::Vector{T},u::Vector{T},lcdf::T,ucdf::T,tp::T,logtp::T) where {T <: Real}
        new{typeof(d),value_support(typeof(d)),T}(d,l,u,lcdf,ucdf,tp,logtp)
    end
end
