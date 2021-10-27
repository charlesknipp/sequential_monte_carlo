using LinearAlgebra,Statistics,Random,Distributions
using ProgressMeter

# import helper functions
include("src/density_tempered_smc.jl")
include("src/dynamic_model.jl")
include("src/particle_filter.jl")


########################## TESTING BLOCK 1 ##########################

# it is good convention to set a seed for testing
Random.seed!(1234)

# recall: set A = .8 instead of an edge case
sims = NDLM(0.8,1.0,1.0,1.0)
_,y = simulate(200,sims)

# guess on the true parameters to see if this works...
guess = [0.5,0.5,.81,.81]
θ,ξ = densityTemperedSMC(200,65,100,y,guess)
map(i -> mean(θ[length(θ)][i,:]),1:4)

#####################################################################



########################## TESTING BLOCK 2 ##########################

function bootstrapFilter(n::Int64,y::Vector{Float64},model::NDLM)
    T = length(y)

    xs = zeros(Float64,T,n)
    qs = zeros(Float64,T,3)
    Z  = zeros(Float64,T)

    x = rand(Normal(model.x0,sqrt(model.Σ0)),n)

    for t in 1:T
        # propogate forward in time
        x = (model.A)*x + rand(Normal(0,sqrt(model.Q)),n)
        
        # calculate likelihood that y fits the simulated distribution
        d  = Normal.((model.B)*x,sqrt(model.R))
        wt = logpdf.(d,y[t])

        # normalize weights and resample
        w = exp.(wt.-maximum(wt))
        w = w/sum(w)
        κ = wsample(1:n,w,n)
        x = x[κ]

        # store the normalizing constant and sample
        Z[t] = mean(wt.-maximum(wt))
        xs[t,:] = x
        qs[t,:] = quantile(x,[.25,.50,.75])
    end

    return Z,xs,qs
end


function binarySearch(B,ph,S1,ξ)
    ξh = 1.0
    S2 = 0.0

    # the idea here is to cut ξ in half until ESS ≈ B
    while true
        # see equation (8) from Duan & Fulop and note ph is log valued
        sh = (ξh-ξ)*ph

        Sh = S1.+sh
        Sh = exp.(Sh.-maximum(Sh))
        S2 = Sh/sum(Sh)

        # if ESS ≈ B break, else ξ = .5*ξ
        ESS(S2)-B < 1.0 ? ξh *= .5 : break
    end

    return ξh,log.(S2)
end

# previous iteration of binary search
function binarySearch(B,ph,S1,ξ)
    ξh  = 1.0
    S2 = 0.0

    ph = exp.(ph)
    S1 = exp.(S1)

    # the idea here is to cut ξ in half until ESS ≈ B
    while true
        # see equation (8) from Duan & Fulop
        sh = ph.^(ξh-ξ)

        Sh = S1.*sh
        S2 = Sh/(Sh'*sh)

        # if ESS ≈ B break, else ξ = .5*ξ
        ESS(S2)-B < 1.0 ? ξh *= .5 : break
    end

    return ξh,log.(Sh2)
end


function gridSearch(B,ph,S1,ξ)
    ξh  = Vector(0.0:0.001:1.0)
    nξ  = length(ξh)
    S2 = zeros(Float64,nξ,length(S1))

    search_space = zeros(Float64,nξ)

    for i in 1:nξ
        # see equation (8)
        sh = (ξh[i]-ξ)*ph

        Sh = S1.+sh
        Sh = exp.(Sh.-maximum(Sh))
        S2[i,:] = Sh/sum(Sh)

        search_space[i] = abs(ESS(S2[i,:])-B)
    end
    
    _,idx = findmin(search_space)

    return ξh[idx],S2[idx,:]
end

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
ξ = zeros(Float64,P)

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
    S[1][i] = -log(N)
end

# store standard deviation & mean of θ[1] for initialization
mθ = [0,map(i -> mean(θ[1][i,:]),1:k)]
σθ = [0,map(i -> std(θ[1][i,:]),1:k)]

# @distributed to run in parallel (is not currently efficient)

# perform a binary search to find ε s.t. ESS ≈ N/2
l=3

ξ[l],S[l] = gridSearch(N/2,ph[l-1],S[l-1],ξ[l-1])
θ[l] = hcat([wsample(θ[l-1][i,:],exp.(S[l]),N) for i in 1:k]...)'

if ESS(exp.(S[l])) < N/2; S[l] = [-log(N) for _ in 1:N] end
    
mθ[1] = mθ[2]
σθ[1] = σθ[2]

mθ[2] = map(i -> mean(θ[l][i,:]),1:k)
σθ[2] = map(i -> std(θ[l][i,:]),1:k)

# this can be parallelized for i ϵ 1,..,N
for i in 1:N
    θi = θ[l][:,i]

    # TODO: make ph log valued and find ph = sum of bf[1]
    mod_i = NDLM(θi[1],θi[2],θi[3],θi[4])
    ph[l][i] = sum(bootstrapFilter(M,y,mod_i)[1])

    # MH step (not encased in function since it's iterated)
    γ = ξ[l]*ph[l][i]+logpdfPrior(θ[l][:,i],θ₀)
    γ = γ-(ξ[l]*ph[l-1][i]+logpdfPrior(θ[l-1][:,i],θ₀))

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