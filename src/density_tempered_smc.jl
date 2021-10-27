using LinearAlgebra,Statistics,Random,Distributions
using ProgressMeter

include("dynamic_model.jl")
include("particle_filter.jl")


"""
    ESS(w)

Computes the effective sample size of a given model using the weights.
For Fulop & Duan, the weights are the vector S[l] at dist. l, while
particle filters use the vector of weights w.
"""
function ESS(w::Vector{Float64})
    return 1/sum(w.^2)
end


"""
    binarySearch(B,ph[l-1],S[l-1],ξ[l-1])

Searches the unit interval for the optimal step size `ξ` by way of a 
bisection method as opposed to a grid search.
"""
function binarySearch(B,ph,S1,ξ,ε=1.0,max_iters=1000)
    ξu,ξl,ξh = 1.0,0.0,0.5

    S2 = 0.0
    i  = 1

    # the idea here is to cut ξ in half until ESS ≈ B
    while i <= max_iters
        ξh = .5*(ξu+ξl)

        # see equation (8) from Duan & Fulop and note ph is log valued
        sh = (ξh-ξ)*ph

        Sh = S1.+sh
        Sh = exp.(Sh.-maximum(Sh))
        S2 = Sh/sum(Sh)

        N_eff = ESS(S2)
        i += 1

        if N_eff-B < ε
            ξu = ξh*.5
        elseif N_eff-B > ε
            ξl = ξh*.5
        else
            break
        end
    end

    return ξh,log.(S2)
end


"""
    gridSearch(B,ph[l-1],S[l-1],ξ[l-1])

Searches the unit interval for the optimal step size `ξ` by way of a
grid search. This is likely more compuationally intense than the other
method, and thus is least preferred; although this is the chosen
routine performed in the paper.

This is curently unused, and only serves as a placeholder until I know
for sure whether it makes a difference
"""
function gridSearch(B,ph,S1,ξ1,step=0.005)
    nξ = length(ξh)
    nS = length(S1)

    ξ2 = Vector(0.0:step:1.0)
    S2 = zeros(Float64,nξ,nS)
    
    ξ = ξ1
    S = S1

    for i in 1:nξ
        # see equation (8) from Duan & Fulop
        sh = (ξ2[i]-ξ1)*ph

        Sh = S1.+sh
        Sh = exp.(Sh.-maximum(Sh))

        S2[i,:] = Sh/sum(Sh)
        N_eff   = ESS(S2[i,:])
        
        if N_eff >= B
            # pick ξ such that ESS is the upper limit of B
            ξ,S = ξ[i-1],S[i-1]
        else
            # for the case that ξ == 1.0
            ξ,S = ξ[i],S[i]
        end
    end
    
    return ξ,S
end


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
    densityTemperedSMC(N,M,P,y,θ₀)

See the paper by Duan and Fulop for details on this algorithm and it's
construction. This is only my adaptation of their research which is a
work in progress in its current state.

- N ≡ # of parameter particles
- M ≡ # of state particles
- P ≡ # of distributions

```
sims = NDLM(0.8,1.0,1.0,1.0)
_,y = simulate(200,sims)

guess = [0.5,0.5,1.0,1.0]
θ = densityTemperedSMC(N=200,M=1000,P=100,y,guess)
```
```julia-repl
Progress: 100%|███████████████████████████████████████████████████████|  ETA: 0:00:0
> map(i -> mean(θ[length(θ)][i,:]),1:4)
44-element Vector{Float64}:
  -0.187400150014130
   0.287833590591023
   1.111236096158105
   1.733121643184528
```

Clearly this needs some work, but in the meantime it compiles without
error. I think issues may arise with rejuvination, along the MH step,
or the binary search.
"""
function densityTemperedSMC(N::Int64,M::Int64,P::Int64,y::Vector{Float64},θ₀::Vector{Float64})
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

        # see equation (4) from Duan & Fulop and recall ph is log valued
        mod_i = NDLM(θi[1],θi[2],θi[3],θi[4])
        ph[1][i] = sum(bootstrapFilter(M,y,mod_i)[1])

        # weights evenly for l=1
        S[1][i] = -log(N)
    end

    # store standard deviation & mean of θ[1] for initialization
    mθ = [0,map(i -> mean(θ[1][i,:]),1:k)]
    σθ = [0,map(i -> std(θ[1][i,:]),1:k)]

    # track the ETA for the algorithm
    pbar = Progress(P,1)

    # @distributed to run in parallel (is not currently efficient)
    for l in 2:P
        next!(pbar)

        # perform a binary search to find ξ s.t. ESS ≈ N/2
        ξ[l],S[l] = gridSearch(N/2,ph[l-1],S[l-1],ξ[l-1])
        θ[l] = hcat([wsample(θ[l-1][i,:],exp.(S[l]),N) for i in 1:k]...)'

        # rejuvination if ESS < B
        if (ESS(exp.(S[l])) < N/2) S[l] = [-log(N) for _ in 1:N] end
        
        mθ[1] = mθ[2]
        σθ[1] = σθ[2]

        mθ[2] = map(i -> mean(θ[l][i,:]),1:k)
        σθ[2] = map(i -> std(θ[l][i,:]),1:k)

        # this can be parallelized for i ϵ 1,..,N
        for i in 1:N
            θi = θ[l][:,i]

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
    end

    finish!(pbar)
    return θ,ξ
end