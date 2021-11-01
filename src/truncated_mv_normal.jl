using Distributions,LinearAlgebra


"""
    randTruncatedMvNormal(N,μ,Σ,l,u)

Generates a random sample of θs given parameters μ and Σ. We use a
truncated normal distribution such that A,B ∈ (-1,1) and Q,R ∈ (0,Inf).
"""
function randTruncatedMvNormal(N::Int64,μ::Vector{Float64},Σ::Matrix{Float64},l,u)
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

        x[i,:] = quantile(Normal(),prob)*condΣ .+ condμ
    end

    return x
end


"""
    logpdfTruncatedMvNormal(proposal,μ,Σ,l,u)

Calculates the log density at θ given parameters μ and Σ. We use a
truncated normal distribution such that A,B ∈ (-1,1) and Q,R ∈ (0,Inf).
"""
function logpdfTruncatedMvNormal(proposal::Vector{Float64},μ::Vector{Float64},Σ::Matrix{Float64},l,u)
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

        p1 = logpdf(Normal(condμ,sqrt(condΣ)),proposal[i])
        p2 = log(cdf(Normal(),u[i])-cdf(Normal(),l[i]))
        logprob += (p1-p2)

        x[i] = proposal[i]
    end

    return logprob
end
