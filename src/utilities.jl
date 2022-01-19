export randTruncatedMvNormal,logpdfTruncatedMvNormal

function mean(vec::Vector{T}) where T <: Number
    # calculate the arithmetic mean of a vector of real numbers
    μ = sum(vec)/length(vec)
    return μ
end

# for the mean over a matrix
function mean(mat::Matrix{T},dim::Int64=1) where T <: Number
    μ = sum(mat,dims=dim)/size(mat,dim)
    return μ
end


# generates N random points of a truncated normal distribution
function randTruncatedMvNormal(
        N::Int64,
        μ::Vector{Float64},
        Σ::Matrix{Float64},
        l::Vector{T},
        u::Vector{T}
    ) where T <: Number

    k = length(μ)
    x = ones(Float64,k,N).*μ

    # perform this N times over; either vectorize or iterate over another loop
    for i in 1:k
        # convert covariance into a block matrix
        Σ11,Σ22 = Σ[i,i],Σ[1:end .!= i,1:end .!= i]
        Σ12,Σ21 = Σ[1:end .!= i,i]',Σ[1:end .!= i,i]

        x2 = x[1:end .!= i,1]
        μ1,μ2 = μ[i],μ[1:end .!= i]

        condΣ = Σ11 - Σ12*inv(Σ22)*Σ21
        condμ = μ1 + Σ12*inv(Σ22)*(x2-μ2)

        lcdf = cdf(Normal(condμ,sqrt(condΣ)),l[i])
        ucdf = cdf(Normal(condμ,sqrt(condΣ)),u[i])
        prob = rand(Uniform(lcdf,ucdf),N)

        x[i,:] = quantile(Normal(),prob)*sqrt(condΣ) .+ condμ
    end

    return x
end


# Calculates the log density at θ given parameters μ and Σ
function logpdfTruncatedMvNormal(
        y::Vector{Float64},
        μ::Vector{Float64},
        Σ::Matrix{Float64},
        l::Vector{T},
        u::Vector{T}
    ) where T <: Number

    k = length(μ)
    x = μ

    logprob = 0

    for i in 1:k
        # convert covariance into a block matrix
        Σ11,Σ22 = Σ[i,i],Σ[1:end .!= i,1:end .!= i]
        Σ12,Σ21 = Σ[1:end .!= i,i]',Σ[1:end .!= i,i]

        x2 = x[1:end .!= i]
        μ1,μ2 = μ[i],μ[1:end .!= i]

        condΣ = Σ11 - Σ12*inv(Σ22)*Σ21
        condμ = μ1 + Σ12*inv(Σ22)*(x2-μ2)

        p1 = logpdf(Normal(condμ,sqrt(condΣ)),y[i])
        p2 = log(cdf(Normal(),u[i])-cdf(Normal(),l[i]))
        logprob += (p1-p2)

        x[i] = y[i]
    end

    return logprob
end