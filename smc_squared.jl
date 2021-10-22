using LinearAlgebra,Statistics,Random,Distributions
using ProgressMeter


# it is good convention to set a seed
Random.seed!(1234)


"""
    pdfPrior(θ,μ,Σ)

Calculates the probability density at θ given parameters μ and Σ.
We use a truncated normal distribution such that A,B ∈ (-1,1) and
Q,R ∈ (0,Inf).
"""
function pdfPrior(θ,μ,Σ=[1.0,1.0,1.0,1.0])
    pA = pdf(TruncatedNormal(μ[1],Σ[1],-1,1),θ[1])
    pB = pdf(TruncatedNormal(μ[2],Σ[2],-1,1),θ[2])
    pQ = pdf(TruncatedNormal(μ[3],Σ[3],0,Inf),θ[3])
    pR = pdf(TruncatedNormal(μ[4],Σ[4],0,Inf),θ[4])

    return prod([pA,pB,pQ,pR])
end


"""
    logpdfPrior(θ,μ,Σ)

Calculates the log density at θ given parameters μ and Σ. We use a
truncated normal distribution such that A,B ∈ (-1,1) and Q,R ∈ (0,Inf).
"""
function logpdfPrior(θ,μ,Σ=[1.0,1.0,1.0,1.0])
    pA = logpdf(TruncatedNormal(μ[1],Σ[1],-1,1),θ[1])
    pB = logpdf(TruncatedNormal(μ[2],Σ[2],-1,1),θ[2])
    pQ = logpdf(TruncatedNormal(μ[3],Σ[3],0,Inf),θ[3])
    pR = logpdf(TruncatedNormal(μ[4],Σ[4],0,Inf),θ[4])

    return sum([pA,pB,pQ,pR])
end

"""
    weightFilter(y,model)

Calculates the weights by way of the bootstrap filter's iterative
process across the time dimension.
"""
function weightFilter(n::Int64,x::Vector{Float64},y::Float64,model::NDLM)
    # propogate forward in time
    x = (model.A)*x + rand(Normal(0,sqrt(model.Q)),n)

    # calculate likelihood that y fits the simulated distribution
    d  = Normal.((model.B)*x,sqrt(model.R))
    wt = logpdf.(d,y)

    # normalize weights
    w = exp.(wt.-maximum(wt))
    w = w/sum(w)

    # store the normalizing constant and resample
    Z = mean(wt.-maximum(wt))
    κ = wsample(1:n,w,n)

    return Z,κ
end


"""
    SMC²(M,y,model)

A method proposed by Chopin (2012) which nputs M = # of θ particles,
N = # of state particles, ...
"""
function SMC²(M::Int64,N::Int64,y::Vector{Float64},θ₀::Vector{Float64})
    θ = zeros(Float64,4,M)
    ω = ones(Float64,M)

    T = length(y)
    Z = zeros(Float64,T)
    a = zeros(Float64,T,N)
    x = zeros(Float64,T,N)

    # pick an initial guess for θ, and make sure Q,R > 0
    θ[1,:] = rand(TruncatedNormal(θ₀[1],1,-1,1),N)
    θ[2,:] = rand(TruncatedNormal(θ₀[2],1,-1,1),N)
    θ[3,:] = rand(TruncatedNormal(θ₀[3],1,0,Inf),N)
    θ[4,:] = rand(TruncatedNormal(θ₀[4],1,0,Inf),N)

    for i in 1:M
        modᵢ = NDLM(θ[1,i],θ[2,i],θ[3,i],θ[4,i])

        x[1,i] = rand(Normal(),1)
        x[1,i] = (modᵢ.A)*x[1,i] + rand(Normal.(0,sqrt(modᵢ.Q)),1)
    end
    
    for t in 1:T
        Z[t],a[t] = weightFilter(M,x[t,:],y[t],modᵢ)
        x[t] = x[a[t]]

        ω = Z[t]*ω

        # degenerecy condition
    end

    return θ[t]
end