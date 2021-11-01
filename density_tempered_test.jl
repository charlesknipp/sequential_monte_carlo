using LinearAlgebra,Statistics,Random,Distributions
using ProgressMeter

cd("src")

# import helper functions
include("dynamic_model.jl")
include("particle_filter.jl")
include("truncated_mv_normal.jl")
include("density_tempered_smc.jl")

########################## TESTING BLOCK 1 ##########################

# it is good convention to set a seed for testing
Random.seed!(1234)

# recall: set A = .8 instead of an edge case
sims = NDLM(0.3,0.4,1.0,1.0)
_,y = simulate(200,sims)

# guess on the true parameters to see if this works...
guess = [0.5,0.5,.81,.81]
θ,ξ = densityTemperedSMC(200,65,100,y,guess)
map(i -> mean(θ[length(θ)][i,:]),1:4)

#####################################################################


########################## TESTING BLOCK 2 ##########################

θ₀ = guess
N = 200
M = 65
P = 100

k = length(θ₀)
lb = zeros(Float64,k)
ub = ones(Float64,k)

# define the initial standard deviation by I(k)
Σ₀ = Matrix{Float64}(I,k,k)

# initialize sequences (eventually replace 4 with length(priors))
θ  = [zeros(Float64,k,N) for _ in 1:P]
ph,S = ([zeros(Float64,N) for _ in 1:P] for _ in 1:2)
ξ = zeros(Float64,P)

# pick an initial guess for θ, and make sure Q,R > 0
θ[1] = randTruncatedMvNormal(N,θ₀,Σ₀,zeros(k),ones(k))

# initialization can be parallelized for i ϵ 1,..,N
for i in 1:N
    θi = θ[1][:,i]

    # see equation (4) from Duan & Fulop and recall ph is log valued
    mod_i = NDLM(θi[1],θi[2],θi[3],θi[4])
    ph[1][i] = sum(bootstrapFilter(M,y,mod_i)[1])

    # weights evenly for l=1
    S[1][i] = -log(N)
end

# store standard deviation & mean of θ[1] for initialization
mθ = [0,mean(θ[1],dims=2)]
σθ = [0,cov(θ[1],dims=2)]

l = 2

# perform a grid search to find max(ξ) such that ESS < N/2
ξ[l],S[l] = gridSearch(N/2,ph[l-1],S[l-1],ξ[l-1])
θ[l] = hcat([wsample(θ[l-1][i,:],exp.(S[l]),N) for i in 1:k]...)'

# rejuvinate
S[l] = [-log(N) for _ in 1:N]

mθ[1] = mθ[2]
σθ[1] = σθ[2]

mθ[2] = mean(θ[l],dims=2)
σθ[2] = cov(θ[l],dims=2)

# this can be parallelized for i ϵ 1,..,N
for i in 1:N
    θi = θ[l][:,i]

    mod_i = NDLM(θi[1],θi[2],θi[3],θi[4])
    ph[l][i] = sum(bootstrapFilter(M,y,mod_i)[1])

    # MH step (not encased in function since it's iterated)
    γ = ξ[l]*ph[l][i]+logpdfTruncatedMvNormal(θ[l][:,i],θ₀,Σ₀,lb,ub)
    γ = γ-(ξ[l]*ph[l-1][i]+logpdfTruncatedMvNormal(θ[l-1][:,i],θ₀,Σ₀,lb,ub))

    h = logpdfTruncatedMvNormal(θ[l-1][:,i],mθ[2],σθ[2],lb,ub)
    h = h-logpdfTruncatedMvNormal(θ[l][:,i],mθ[1],σθ[1],lb,ub)

    α = min(1,exp(γ+h))
    u = rand()

    if u > α
        θ[l][:,i] = θ[l-1][:,i]
    end
end

finish!(pbar)