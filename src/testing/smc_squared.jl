using Distributions
using LinearAlgebra

function wcov(X::Matrix{Float64},μ::Vector{Float64},w::Vector{Float64})
    k,_ = size(X)
    Σi  = zeros(Float64,k,k)

    for i in 1:length(w)
        Σi += w[i]*((X[:,i] - μ)*(X[:,i] - μ)')
    end

    return Σi /sum(w)
end


function ESS(x::Vector{Float64})
    ESS = sum(x)^2 / sum(x.^2)
    if ESS == NaN; ESS = 0.0 end

    return ESS
end


function randomWalkMH(ω::Vector{Float64},x::Matrix{Float64},c::Float64=.5)
    k,n = size(x)

    μ = [sum(ω.*x[i,:])/sum(ω) for i in 1:k]
    Σ = wcov(x,μ,ω)

    # defined by the prior, please get rid of this eventually
    low = [-1.0,-1.0,0.0,0.0]
    up  = [1.0,1.0,Inf,Inf]

    return randTruncatedMvNormal(n,x,c*Σ,low,up)
end


function SMC²(M::Int64,N::Int64,y::Vector{Float64},θ₀::Vector{Float64})
    # note: M = # of θ particles
    #       N = # of x particles
    k = size(θ)[1]

    # θ = randTruncatedMvNormal(M,θ₀,Matrix{Float64}(I,k,k),lb,ub)
    ω = ones(Float64,M)

    # initialize unobserved arrays
    x = zeros(Float64,T,M,N)
    a = zeros(Float64,T,M,N)

    for m in 1:M
        # sample θ^{m} from p(θ) and set ω^{m} ← 1
        θₘ = randTruncatedMvNormal(1,θ₀,Matrix{Float64}(I,k,k),lb,ub)   # prior
        ωₘ = ones(Float64,M)
        modₘ = NDLM(θₘ[1],θₘ[2],θₘ[3],θₘ[4])

        # for time t = 1,...,T
        for t in 1:T
            # for each θ^{m} perform iteration t of a PF
            if t == 1
                # sample independently x_{1}^{1:Nx,m} from ψ_{1,θ^{m}}
                x0 = rand(Normal(),N)
                x[t,m,:] = (modₘ.A)*x0 .+ rand(Normal.(0,sqrt(modₘ.Q)),N) # AR process
            else
                # sample a_{t-1}^{1:Nx,m} and x_{t}^{1:Nx,m} from ψ_{t,θ^{m}}
                a[t-1,m,:] = wsample(1:N,Wₜ,N)
                x[t,m,:]   = (modₘ.A)*x[t-1,m,:] .+ rand(Normal.(0,sqrt(modₘ.Q)),N)
            end

            # calculate weights: w
            gₜ = Normal.((modₘ.B)*x[t,m,:],sqrt(modₘ.R))
            wₜ = logpdf.(gₜ,y[t])

            # normalize weights: W
            Wₜ = exp.(wₜ.-maximum(wₜ))
            Wₜ = Wₜ/sum(Wₜ)

            # find p̂(y_{t}|y_{1:t-1},θ^{m}) = E[w_{t,θ}(x_{t-1}^{a_{t-1}^{n,m},m})]
            log_p̂ = log(mean(exp.(wₜ.-maximum(wₜ)))) + maximum(wₜ)

            # update the weights (not log valued)
            ωₘ = ωₘ*exp.(log_p̂)

            if ESS(ωₘ) ≤ M/2
                θ = randomWalkMH(ωₘ,θₘ,0.5)

                for i in 1:10
                    # iterate the process of picking α and u
                end
            end
        end
    end
end