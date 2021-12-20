using LinearAlgebra,Statistics,Random,Distributions
using ProgressMeter

include("dynamic_model.jl")
include("particle_filter.jl")


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

    # store the normalizing constant
    Z = mean(wt.-maximum(wt))
    logZ = log(mean(exp.(wt.-maximum(wt)))) + maximum(wt)

    # resample
    κ = wsample(1:n,w,n)

    return Z,logZ,κ
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
    prior(μ,n)

Generates a random sample of n particles for the parameters in a normal
dynamic linear model.
"""
function prior(μ,n)
    # make a function to sample from prior...
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

    a = zeros(Float64,T,M,N)
    x = zeros(Float64,T,M,N)

    # pick an initial guess for θ, and make sure Q,R > 0
    θ[1,:] = rand(TruncatedNormal(θ₀[1],1,-1,1),M)
    θ[2,:] = rand(TruncatedNormal(θ₀[2],1,-1,1),M)
    θ[3,:] = rand(TruncatedNormal(θ₀[3],1,0,Inf),M)
    θ[4,:] = rand(TruncatedNormal(θ₀[4],1,0,Inf),M)

    for i in 1:M
        modᵢ = NDLM(θ[1,i],θ[2,i],θ[3,i],θ[4,i])

        x[1,i,:] = rand(Normal(),N)
        x[1,i,:] = (modᵢ.A)*x[1,i,:] .+ rand(Normal.(0,sqrt(modᵢ.Q)),N)
    end

    for t in 1:T
        Z[t],_,a[t,i,:] = weightFilter(M,x[t,i,:],y[t],modᵢ)
        x[t,i,:] = x[a[t,i,:]]

        ω = Z[t]*ω

        # degenerecy condition

        # MH step
        T = MvNormal([0,0,0,0],I(4))    # not sure what to set T() as

    end

    return θ[t]
end


########################## TESTING BLOCK 1 ##########################

M = 20
N = 10
T = 10

θ = zeros(Float64,4,M)
x = zeros(Float64,T,M,N)

θ₀ = [0.6,1.0,1.0,1.0]

# pick an initial guess for θ, and make sure Q,R > 0
θ[1,:] = rand(TruncatedNormal(θ₀[1],1,-1,1),M)
θ[2,:] = rand(TruncatedNormal(θ₀[2],1,-1,1),M)
θ[3,:] = rand(TruncatedNormal(θ₀[3],1,0,Inf),M)
θ[4,:] = rand(TruncatedNormal(θ₀[4],1,0,Inf),M)

for i in 1:M
    modᵢ = NDLM(θ[1,i],θ[2,i],θ[3,i],θ[4,i])
    xᵢ   = rand(Normal(),N)

    x[1,i,:] = (modᵢ.A)*xᵢ .+ rand(Normal.(0,sqrt(modᵢ.Q)),N)
end

#####################################################################
