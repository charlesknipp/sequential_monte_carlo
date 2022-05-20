using LinearAlgebra,StaticArrays,Distributions,Random,LoopVectorization

############################### helper functions ###############################

function sum_all_but(x,i)
    # not sure why this is...
    x[i] -= 1
    xsum = sum(x)
    x[i] += 1

    return xsum
end

tosvec(x) = reinterpret(SVector{length(x[1]),Float64},reduce(hcat,x))[:] |> copy

# define wcov() and wmean()

################################################################################

abstract type AbstractSSM end
abstract type ModelParameters end
abstract type AbstractFilter end


struct StateSpaceModel{F,G,F0} <: AbstractSSM
    # define general structure using functions
    transition::F
    observation::G
    initial_dist::F0
end

struct LinearGaussian{T<:Real} <: ModelParameters
    A::Matrix{T}
    B::Matrix{T}

    Q::Matrix{T}
    R::Matrix{T}

    μ0::Array{T}
    Σ0::Matrix{T}
end

function StateSpaceModel(params::LinearGaussian)
    # import parameters
    A,B = params.A,params.B
    Q,R = params.Q,params.R

    f(xt) = MvNormal(A*xt,Q)
    g(xt) = MvNormal(B*xt,R)

    f0 = MvNormal(params.μ0,params.Σ0)

    return StateSpaceModel(f,g,f0)
end

dims(mod::LinearGaussian) = (size(mod.A,1),size(mod.B,1))

# old method using model type
function simulate(rng::AbstractRNG,mod::AbstractSSM,T::Int64)
    #dx,dy = dims(mod)
    dx,dy = length(mod.initial_dist),length(mod.initial_dist) # just for testing

    x = Matrix{Float64}(undef,dx,T)
    y = Matrix{Float64}(undef,dy,T)
    
    x[:,1] = rand(rng,mod.initial_dist)

    # try experimenting with views and inbounds
    for t in 1:T-1
        x[:,t+1] = rand(rng,mod.transition(x[:,t]))
    end

    for t in 1:T
        y[:,t] = rand(rng,mod.observation(x[:,t]))
    end

    return x,y
end

# sample using filter object
function simulate(pf::AbstractFilter,T::Int)
    mod = pf.model

    x = Vector{Vector{Float64}}(undef,T)
    y = similar(x)

    x[1] = rand(pf.rng,mod.initial_dist)

    for t = 1:T-1
        y[t] = rand(pf.rng,mod.observation(x[t]))
        x[t+1] = rand(pf.rng,mod.transition(x[t]))
    end

    y[T] = rand(pf.rng,mod.observation(x[T]))
    return x,y
end


struct Particles{PT,FT}
    x::Vector{PT}
    xprev::Vector{PT}

    logw::Vector{FT}
    w::Vector{FT}
    maxw::Base.RefValue{FT}

    # a is the index and t is the time
    a::Vector{Int64}
    t::Base.RefValue{Int}
end

Particles(N::Integer) = Particles(
    [zeros(N)],
    [zeros(N)],
    fill(-log(N),N),
    fill(1/N,N),
    Ref(0.),
    collect(1:N),
    Ref(1)
)

Base.length(p::Particles) = length(p.x)
ESS(p::Particles) = 1.0/sum(abs2,p.w)

function reset_weights!(p)
    N = length(p)
    fill!(p.logw,log(1/N))
    fill!(p.w,1/N)
    p.maxw[] = 0
end

reset_weights!(pf::AbstractFilter) = reset_weights!(pf.state)


struct ParticleFilter{ST} <: AbstractFilter
    model::StateSpaceModel
    state::ST

    resample_threshold::Float64
    rng::AbstractRNG
end

function ParticleFilter(N::Integer,model,B=0.1,rng=Random.GLOBAL_RNG)
    mod_type = eltype(model.initial_dist)
    dim_x = length(model.initial_dist)

    xprev = Vector{SVector{dim_x,mod_type}}([rand(rng,model.initial_dist) for n=1:N])
    x = deepcopy(xprev)

    logw = fill(-1*log(N),N)
    w = fill(1/N,N)

    s = Particles(x,xprev,logw,w,Ref(0.),collect(1:N),Ref(1))

    return ParticleFilter(model,s,B,rng)
end


Base.@propagate_inbounds function reweight!(pf::ParticleFilter,y)
    logw = pf.state.logw
    dist = pf.model.observation
    any(ismissing,y) && return logw

    if dist isa UnivariateDistribution && length(y) == 1
        for i = 1:length(pf.state)
            # not sure if logw += logpdf(⋅) or logw = logpdf(⋅)
            logw[i] = logpdf(dist(pf.state.x[i][1]),y[1])
        end
    else
        for i = 1:length(pf.state)
            # not sure if logw += logpdf(⋅) or logw = logpdf(⋅)
            logw[i] = logpdf(dist(pf.state.x[i]),y)
        end
    end

    return logw
end

# normalizes weights and updates particle cloud (too complicated)
function logsumexp!(logw,w,maxw=Ref(zero(eltype(logw))))::eltype(logw)
    offset,maxind = findmax(logw)
    logw .-= offset

    # normalize new weights
    LoopVectorization.vmap!(exp,w,logw)
    sumw   = sum_all_but(w,maxind)
    w    .*= 1/(sumw+1)
    logw .-= log1p(sumw)

    # adjusted maximum log weight
    maxw[] += offset

    return log1p(sumw) + maxw[] - log(length(logw))
end

@inline logsumexp!(p) = logsumexp!(p.logw,p.w,p.maxw)
@inline logsumexp!(pf::AbstractFilter) = logsumexp!(pf.state)


# resampling required
Base.@propagate_inbounds function propagate!(pf::ParticleFilter,a::Vector{Int})
    s = pf.state
    transition = pf.model.transition
    x,xp = s.x,s.xprev

    vec_type = eltype(x)
    d_dims   = length(vec_type)
    xprop    = zeros(d_dims)

    for i = eachindex(x)
        x[i] = vec_type(rand!(pf.rng,transition(xp[a[i]]),xprop))
    end

    return x
end

# no need for resampling
Base.@propagate_inbounds function propagate!(pf::ParticleFilter)
    s = pf.state
    transition = pf.model.transition
    x,xp = s.x,s.xprev

    vec_type = eltype(x)
    d_dims   = length(vec_type)
    xprop    = zeros(d_dims)

    for i = eachindex(x)
        x[i] = vec_type(rand!(pf.rng,transition(xp[i]),xprop))
    end

    return x
end

index(pf::AbstractFilter) = pf.state.t[]


function correct!(pf,y)
    # calculates log weights
    reweight!(pf,y)

    # normalizes weights and finds the likelihood
    ll = logsumexp!(pf.state)

    return ll
end

function predict!(pf)
    particles = pf.state
    N = length(particles)

    # rethink this in terms of particle types/filter types
    if ESS(particles) < pf.resample_threshold*N
        a = wsample(pf.rng,1:N,particles.w,N)
        propagate!(pf,a)
        reset_weights!(particles)
    else # Resample not needed
        particles.a .= 1:N
        propagate!(pf)
    end

    # move forward in time
    copyto!(particles.xprev,particles.x)
    pf.state.t[] += 1
end

function update!(pf::AbstractFilter,y)
    ll = correct!(pf,y)
    predict!(pf)

    return ll
end

function reset!(pf::AbstractFilter)
    particles = pf.state

    for i = eachindex(particles.xprev)
        particles.xprev[i] = rand(pf.rng,pf.model.initial_dist)
        particles.x[i] = copy(particles.xprev[i])
    end

    fill!(particles.logw,-log(length(particles)))
    fill!(particles.w,1/length(particles))

    pf.state.t[] = 1
end

function log_likelihood(pf::AbstractFilter,y)
    reset!(pf)

    return sum(x -> update!(pf,x),y)
end

# θ -> p(y|θ)p(θ) with filter_from_parameters = f(θ::Vector)
function log_likelihood_fun(filter_from_parameters,prior,y)
    pf = nothing

    function (θ)
        # check whether priors are same dims as proposal
        pf === nothing && (pf=filter_from_parameters(θ))
        length(θ) == length(prior) || throw(ArgumentError("Input must have same length as priors"))

        # calculate the likelihood per each proposal
        ll = logpdf(prior,θ)
        isfinite(ll) || return eltype(θ)(-Inf)
        
        pf = filter_from_parameters(θ,pf)
        return ll + log_likelihood(pf,y)
    end
end

# assuming y is a vector of vectors
function bootstrap_filter(N,model,y,T=length(y),B=0.1,rng=Random.GLOBAL_RNG)
    pf = ParticleFilter(N,model,B,rng)

    for t in 1:T
        update!(pf,y[t])
    end

    return pf
end

test_model_1 = LinearGaussian(
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    zeros(1),Matrix(1.0I(1))
)

test_model_2 = LinearGaussian(
    Matrix(1.0I(2)),Matrix(1.0I(2)),
    Matrix(1.0I(2)),Matrix(1.0I(2)),
    zeros(2),Matrix(1.0I(2))
)

ssm1 = StateSpaceModel(test_model_1)
ssm2 = StateSpaceModel(test_model_2)

pf1 = ParticleFilter(100,ssm1)
pf2 = ParticleFilter(100,ssm2)

x1,y1 = simulate(pf1,50)
x2,y2 = simulate(pf2,50)

# > @btime bootstrap_filter(100,$ssm1,$y1)
#   1.910 ms (20271 allocations: 739.03 KiB)

##################################### SMC² #####################################

struct SMC²{ST,PD,SSM,K}
    params::ST
    prior::PD
    model::SSM
    mcmc_kernel::K

    N::Int
    chain_len::Int

    resample_threshold::Float64
    rng::AbstractRNG
end

# not 100% convinced this is the best way
function SMC²(M::Int,N::Int,mcmc_kernel,θ0,prior,B,chain_len,rng=Random.GLOBAL_RNG)
    mod_type = eltype(prior(θ0))
    dimθ = length(prior(θ0))

    θprev = Vector{SVector{dimθ,mod_type}}([rand(rng,prior(θ0)) for m=1:M])
    θ = deepcopy(θprev)

    logw = fill(-log(M),M)
    w = fill(1/M,M)

    θ = Particles(θ,θprev,logw,w,Ref(0.),collect(1:M),Ref(1))

    return SMC²(θ,prior(θ0),model,mcmc_kernel,N,chain_len,B,rng)
end

# quick and dirty approach, but not the right one
function random_walk(θ0)
    !any(iszero.(θ0)) || throw(ArgumentError("Initial value must be non-zero"))
    return θ -> θ .+ rand(MvNormal(0.1abs.(θ0)))
end

# expand to work for θ particles
function metropolis(logprob,chain_len,θ0,mcmc_kernel=random_walk(θ0))
    # here logprob is the smc kernel: Xt.p[m].logμ + logpdf(pθ,θ.x[m])
    θ  = Vector{typeof(θ0)}(undef,chain_len)
    ll = Vector{Float64}(undef,chain_len)

    θ[1]  = θ0
    # develop an SMC² object to store the log probabilities
    ll[1] = logprob(θ0)

    for i = 2:N
        θi = mcmc_kernel(θ[i-1])
        lli = logprob(θi)
        if rand() < exp(lli-ll[i-1])
            θ[i] = θi
            ll[i] = lli
        else
            θ[i] = θ[i-1]
            ll[i] = ll[i-1]
        end
    end

    return θ[chain_len]
end

function rejuvenate!(smc²::SMC²,logprob)
    θ = smc².params

    for i = eachindex(smc².params.x)
        θ.x[i] = metropolis(logprob,smc².chain_len,θ.x[i],smc².mcmc_kernel)
    end
end

@inline logsumexp!(smc²::SMC²) = logsumexp!(smc².params)


function update_importance!(smc²::SMC²,y)
    function bootstrap_filter(θ,pf=nothing)
        mod = smc².model(θ)
        return ParticleFilter(smc².N,mod)
    end

    θ = smc².params
    logZ = log_likelihood_fun(bootstrap_filter,smc².prior,y[1:θ.t[]])

    # reweight
    for i in eachindex(θ.x)
        smc².params.logw[i] += logZ(θ.x[i])
    end

    logsumexp!(smc²)    # normalize the weights

    if ESS(smc².params) < smc².resample_threshold*length(θ)
        rejuvenate!(smc²,bootstrap_filter)
        reset_weights!(smc².params)
    end
end

function reset!(smc²::SMC²)
    θ = smc².params

    for i = eachindex(θ.xprev)
        θ.xprev[i] = rand(smc².rng,smc².prior)
        θ.x[i] = copy(θ.xprev[i])
    end

    fill!(θ.logw,-log(length(θ)))
    fill!(θ.w,1/length(θ))

    pf.state.t[] = 1
end

#=
    Construct a function that takes θ and generates a model likelihood like the
    LLPF package does for PMMH.
=#

model(θ) = StateSpaceModel(LinearGaussian(
    Matrix(θ[1]I(1)),Matrix(θ[2]I(1)),
    Matrix(exp(θ[3])I(1)),Matrix(exp(θ[4])I(1)),
    zeros(1),Matrix(1.0I(1))
))

θ1 = [
    Matrix(0.8I(1)),Matrix(0.5I(1)),
    Matrix(1.0I(1)),Matrix(1.0I(1))
]

θ1 = [0.8,0.5,1.0,1.0]

# finish the prior formatting/structure
prior_func(θ) = product_distribution([
    TruncatedNormal(θ[1],1.0,-1.0,1.0),
    TruncatedNormal(θ[2],1.0,-1.0,1.0),
    Normal(0.0,2.0),
    Normal(0.0,2.0)
])

# ALMOST DONE!!! Run this with the debugger to check the distribution being constructed

smc2 = SMC²(20,100,random_walk(θ1),θ1,prior_func,0.1,2)
update_importance!(smc2,y_test[1])

# test bootstrap filter:
test_model = LinearGaussian(
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    Matrix(1.0I(1)),Matrix(1.0I(1)),
    zeros(1),Matrix(1.0I(1))
)

ssm_test = StateSpaceModel(test_model)

pf_test = ParticleFilter(1000,ssm_test,1.0,MersenneTwister(1234))
x_test,y_test = simulate(pf_test,100)

log_likelihood(pf_test,y_test)

using StatsBase

reset!(pf_test)
for t = eachindex(y_test)
    update!(pf_test,y_test[t])
    xt = reduce(hcat,pf_test.state.x)
    x_mean = mean(xt,weights(pf_test.state.w),2)
    println(@sprintf("simulated: %.5f\tfiltered: %.5f",x_test[t][1],x_mean[1]))
end