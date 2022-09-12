using LinearAlgebra,Distributions,Random,Statistics
using Printf

function simulate(rng::AbstractRNG,T::Int,θ::Vector{Float64})
    x = Vector{Float64}(undef,T)
    y = similar(x)

    x[1] = rand(rng,Normal(0.0,θ[2]))

    for t in 1:T-1
        y[t]   = rand(rng,Normal(x[t],θ[3]))
        x[t+1] = rand(rng,Normal(θ[1]*x[t],θ[2]))
    end

    y[T] = rand(rng,Normal(x[T],θ[3]))

    return x,y
end


function particle_filter(rng::AbstractRNG,N::Int,θ::Vector{Float64},y::Vector{Float64})
    # sample X0 from initial distribution
    T = length(y)
    x = zeros(Float64,N,T)
    w = zeros(Float64,N,T)

    x[:,1] = rand(rng,Normal(0.0,θ[2]),N)

    for t in 1:T
        if t == 1
            w[:,t] = logpdf.(Normal.(x[:,t],θ[3]),y[t])
        else
            norm_w = exp.(w[:,t-1].-maximum(w[:,t-1]))
            norm_w /= sum(norm_w)

            a = wsample(rng,1:N,norm_w,N)
            x[:,t] = rand.(rng,Normal.(θ[1]*x[a,t-1],θ[2]))
            w[:,t] = logpdf.(Normal.(x[:,t],θ[3]),y[t])
        end
    end

    logZ = sum(log.(1/N*sum(exp.(w),dims=1)))
    return x[:,T],w[:,T],logZ
end


simulate(T::Int,θ::Vector{Float64}) = simulate(Random.GLOBAL_RNG,T,θ)
particle_filter(N::Int,θ::Vector{Float64},y::Vector{Float64}) = particle_filter(Random.GLOBAL_RNG,N,θ,y)

function smc²(N::Int,M::Int,y::Vector{Float64},chain_len::Int,prior)
    T = length(y)

    x = zeros(Float64,N,M,T)
    θ = zeros(Float64,length(prior),M,T)

    x_w = zeros(Float64,N,M,T)
    θ_w = zeros(Float64,M,T)

    logZ = zeros(Float64,M,T)
    ess  = zeros(Float64,T)

    transition(θ,x)  = Normal.(θ[1]*x,θ[2])
    observation(θ,x) = Normal.(x,θ[3])

    θ[:,:,1] = rand(prior,M)

    for t in 1:T
        # update x weights  
        for m in 1:M
            if t == 1
                x_w[:,m,t] = logpdf.(observation(θ[:,m,t],x[:,m,t]),y[t])
            else
                θ[:,m,t] = θ[:,m,t-1]

                norm_w = exp.(x_w[:,m,t-1].-maximum(x_w[:,m,t-1]))
                norm_w /= sum(norm_w)

                a = wsample(1:N,norm_w,N)
                x[:,m,t] = rand.(transition(θ[:,m,t],x[a,m,t-1]))
                x_w[:,m,t] = logpdf.(observation(θ[:,m,t],x[:,m,t]),y[t])
            end
        end

        # update θ weights (does not account for underflows)
        if t == 1
            θ_w[:,t]  = sum((1/N)*exp.(x_w[:,:,t]),dims=1)
            logZ[:,t] = log.(θ_w[:,t])
        else
            likelihood = sum((1/N)*exp.(x_w[:,:,t]),dims=1)
            θ_w[:,t]   = θ_w[:,t-1] .* likelihood'
            logZ[:,t]  = logZ[:,t-1] + log.(likelihood')
        end

        # normalize θ weights
        θ_w[:,t] = θ_w[:,t] ./ sum(θ_w[:,t])
        ess[t]   = 1.0/sum(θ_w[:,t].^2)

        # check for particle degeneration
        @printf("\nt = %4d\tess = %3.5f",t,ess[t])
        if ess[t] < M*0.8
            @printf("\t[rejuvenating]")

            # resample
            a = wsample(1:M,θ_w[:,t],M)

            θ[:,:,t]  = θ[:,a,t]
            logZ[:,t] = logZ[a,t]

            x[:,:,t] = x[:,a,t]
            x_w[:,:,t] = x_w[:,a,t]

            # define pmmh movement
            dθ = (2.38^2) / length(prior)
            Σ  = norm(cov(θ[:,:,t]')) < 1e-12 ? 1e-2*I : dθ*cov(θ[:,:,t]') + 1e-10*I

            acc_rate = 0.0

            # rejuvenate
            for m in 1:M
                for _ in 1:chain_len
                    θ_prop = rand(MvNormal(θ[:,m,t],Σ))

                    if insupport(prior,θ_prop)
                        x_prop,w_prop,logZ_prop = particle_filter(N,θ_prop,y[1:t])

                        acc_ratio  = logZ_prop - logZ[m,t]
                        acc_ratio += logpdf(prior,θ_prop) - logpdf(prior,θ[:,m,t])

                        if log(rand()) ≤ minimum([1.0,acc_ratio])
                            θ[:,m,t]  = θ_prop
                            logZ[m,t] = logZ_prop
                
                            x[:,m,t]   = x_prop
                            x_w[:,m,t] = w_prop

                            acc_rate += 1
                        end
                    end
                end

                θ_w[m,t] = 1.0
            end

            @printf("\t acc rate: %1.4f",acc_rate/(M*chain_len))
        end
    end

    print("\n")
    return θ,θ_w
end


###############################################################################

T = 100

θ = [0.5,1.0,1.0]
x,y = simulate(T,θ)

prior = product_distribution([
    TruncatedNormal(0.0,1.0,-1.0,1.0),
    LogNormal(),                        # Normal(0.0,2.0),
    LogNormal()                         # Normal(0.0,2.0)
])

pred,w = smc²(512,100,y,3,prior)

w_norm = w[:,T]/sum(w[:,T])

pred_θ = pred[:,:,T]

expected_θ = sum(
    reduce(hcat,[pred[:,i,T]*w_norm[i] for i in 1:100]),
    dims=2
)
