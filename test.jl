using LinearAlgebra,Statistics,Random,Distributions
using ProgressMeter

include("density_tempered_smc.jl")
include("dynamic_model.jl")
include("particle_filter.jl")


########################## TESTING BLOCK 1 ##########################

# recall: set A = .8 instead of an edge case
sims = NDLM(0.8,1.0,1.0,1.0)
_,y = simulate(200,sims)

# guess on the true parameters to see if this works...
guess = [0.8,1.0,1.0,1.0]
θ = densityTemperedSMC(200,65,100,y,[0.5,0.5,.81,.81])
map(i -> mean(θ[length(θ)][i,:]),1:4)

#####################################################################



########################## TESTING BLOCK 2 ##########################

# recall: set A = .8 instead of an edge case
sims = NDLM(0.8,1.0,1.0,1.0)
_,y = simulate(200,sims)

# guess on the true parameters to see if this works...
θ₀ = [0.5,0.5,.81,.81]
N = 200
P = 100
M = 65
k = length(θ₀)

# initialize sequences (eventually replace 4 with length(priors))
θ  = [zeros(Float64,k,N) for _ in 1:P]
ph,S = ([zeros(Float64,N) for _ in 1:P] for _ in 1:4)
ε = zeros(Float64,P)

# pick an initial guess for θ, and make sure Q,R > 0
θ[1][1,:] = rand(TruncatedNormal(θ₀[1],1,-1,1),N)
θ[1][2,:] = rand(TruncatedNormal(θ₀[2],1,-1,1),N)
θ[1][3,:] = rand(TruncatedNormal(θ₀[3],1,0,Inf),N)
θ[1][4,:] = rand(TruncatedNormal(θ₀[4],1,0,Inf),N)

# initialization can be parallelized for i ϵ 1,..,N
for i in 1:N
    θi = θ[1][:,i]

    # set random movements as ph = normalizing constant
    mod_i = NDLM(θi[1],θi[2],θi[3],θi[4])
    ph[1][i] = sum(bootstrapFilter(M,y,mod_i)[1])

    # weights evenly for l=1
    S[1][i] = 1/N
end

# store standard deviation & mean of θ[1] for initialization
mθ = [0,map(i -> mean(θ[1][i,:]),1:k)]
σθ = [0,map(i -> std(θ[1][i,:]),1:k)]

# @distributed to run in parallel (is not currently efficient)

# perform a binary search to find ε s.t. ESS ≈ N/2
l=2

ε[l],S[l] = binarySearch(N/2,exp.(ph[l-1]),S[l-1],ε[l-1])
θ[l] = hcat([wsample(θ[l-1][i,:],S[l],N) for i in 1:k]...)'
    
mθ[1] = mθ[2]
σθ[1] = σθ[2]

mθ[2] = map(i -> mean(θ[l][i,:]),1:k)
σθ[2] = map(i -> std(θ[l][i,:]),1:k)

# this can be parallelized for i ϵ 1,..,N
for i in 1:N
    θi = θ[l][:,i]

    # TODO: make ph log valued and find ph = sum of bf[1]
    mod_i = NDLM(θi[1],θi[2],θi[3],θi[4])
    ph[l][i] = prod(bootstrapFilter(M,y,mod_i)[1])

    # MH step (not encased in function since it's iterated)
    γ = ε[l]*log(ph[l][i])+logpdfPrior(θ[l][:,i],θ₀)
    γ = γ-(ε[l]*log(ph[l-1][i])+logpdfPrior(θ[l-1][:,i],θ₀))

    h = logpdfPrior(θ[l-1][:,i],mθ[2],σθ[2])
    h = h-logpdfPrior(θ[l][:,i],mθ[1],σθ[1])

    α = min(1,exp(γ+h))
    u = rand()

    if u > α
        θ[l][:,i] = θ[l-1][:,i]
    end
end


θ[2]
map(i -> mean(θ[2][i,:]),1:4)

#####################################################################

# One bug I am having is that the vector `ph` tends to 0, in which `ph` represents the estimated marginal likelihood p_{θ}(y_{1:T}). This is likely due to the normalizing constants also tending to 0, a bug present in the `bootstrapFilter()` function; when calculating the weights Julia only evaluates the log likelihoods, thus I have to convert the weight when calculating the normalizing constant. It isn't clear to me how to find the log normalizing constant, so the problem remains...