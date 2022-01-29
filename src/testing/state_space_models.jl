using Distributions
using LinearAlgebra


# define abstract types for models
abstract type AbstractSSM end
abstract type GaussianSSM end

#=
    If I want to implement non-linear Gaussian models, I would need to
    implement a transition_mean parameter to StateSpaceModel. This
    allows for functions to calculate the mean rather than calculating
    it on simulation
=#

struct StateSpaceModel <: AbstractSSM
    # transition function:  f(xₜ|xₜ₋₁)
    transition::Function
    dim_x::Int64

    # observation function: g(yₜ|xₜ)
    observation::Function
    dim_y::Int64
end

"""
Suppose x₀ and Σ₀ are given such that x₁ ~ N(x₀,Σ₀), we define
linear Gaussian state space models via the following form...
    xₜ = Axₜ₋₁ + εₜ     s.t. εₜ ~ N(0,Q)
    yₜ = Bxₜ + δₜ       s.t. δₜ ~ N(0,R)
"""
struct LinearGaussian <: GaussianSSM

    A::Union{Float64,Matrix{Float64}}
    B::Union{Float64,Matrix{Float64}}

    Q::Union{Float64,Matrix{Float64}}
    R::Union{Float64,Matrix{Float64}}

    # implicitly defined by the constructor
    dim_x::Int64
    dim_y::Int64

    function LinearGaussian(A,B,Q,R)
        dim_x = size(A,1)
        dim_y = size(B,1)

        @assert dim_x==size(A,2)==size(Q,1)==size(Q,2) "dimensions don't match"
        @assert dim_y==size(R,1)==size(R,2) "dimensions don't match"

        @assert issymmetric(Q) && issymmetric(R) "cov mat must be symmetric"
        @assert isposdef(Q) && isposdef(R) "cov mat must be pos def"

        # construct the new object
        new(A,B,Q,R,dim_x,dim_y)
    end
end

function SSM(model::GaussianSSM)
    f(xₜ₋₁,xₜ) = logpdf(Normal(model.A*xₜ₋₁,sqrt(model.Q)),xₜ)
    g(yₜ,xₜ)   = logpdf(Normal(model.B*xₜ,sqrt(model.R)),yₜ)

    StateSpaceModel(f,model.dim_x,g,model.dim_y)
end


function simulate(
        model::GaussianSSM,
        x0::Union{Float64,Vector{Float64}},
        T::Int64
    )::Tuple{Matrix{Float64},Matrix{Float64}}

    n,m = model.dim_x,model.dim_y

    # allocate states/observations
    x = zeros(Float64,n,T)
    y = zeros(Float64,m,T)

    # FIX THIS x[:,1] = [Vector{Float}] is v bad...
    # try using similar to generate a good structure?
    @assert length(x0) == n "dimensions don't match"
    x[:,1] = [x0]

    # choose a kernel K based on dim_x and dim_y
    Kx = (n==1) ? Normal : MvNormal
    Ky = (m==1) ? Normal : MvNormal
    
    # could write it so that univariate distributions eval like so
    # rand.(Ky(model.B*x[:,t],Ref(sqrt(model.R)))), but this kinda
    # sucks since multivariate doesn't work the same...

    for t in 1:(T-1)
        y[:,t]   = model.B*x[:,t] + rand(Ky(0,sqrt(model.R)),m)
        x[:,t+1] = model.A*x[:,t] + rand(Kx(0,sqrt(model.Q)),n)
    end
    
    # last observation
    y[:,T] = model.A*x[:,T] + rand(Ky(0,sqrt(model.R)),m)

    return (x,y)
end

lg  = LinearGaussian(1.0,1.0,1.0,1.0)
hmm = SSM(lg)

x,y = simulate(lg,0.0,100)



# particles must be declared mutable since weights are subject to change
# as we continue to refine the sample
mutable struct Particles
    x::Matrix{Float64}
    logw::Vector{Float64}
end

function Particles(N::Int64,dims::Int64=1)
    particles   = zeros(Float64,dims,N)
    log_weights = fill(-1*log(N),N)

    return Particles(particles,log_weights)
end

function mean(p::Particles)
    k,_ = size(p.x)
    max_x = maximum(p.x,dims=2)
    w = exp.(p.logw.-maximum(p.logw))

    μ = [exp(max_x[i])*sum(w.*exp.(p.x[i,:].-max_x[i]))/sum(w) for i in 1:k]
    return μ
end


mutable struct ParticleSet
    p::Vector{Particles}
end

function ParticleSet(N::Int64,dim::Int64,T::Int64)
    particle_set = [Particles(N,dim) for _ in 1:T]

    return ParticleSet(particle_set)
end

function mean(ps::ParticleSet)
    return [mean(p) for p in ps.p]
end


function ESS(logx::Vector{Float64})
    max_logx = maximum(logx)
    x_scaled = exp.(logx.-max_logx)

    ESS = sum(x_scaled)^2 / sum(x_scaled.^2)
    if ESS == NaN; ESS = 0.0 end

    return ESS
end

function resample(p::Particles,B::Number=Inf)::Tuple{Particles,Float64}
    ess = ESS(p.logw)
    N = size(p.x,2)

    if ess < B*N
        w = exp.(p.logw.-maximum(p.logw))
        w = w/sum(w)
        κ = wsample(1:N,w,N)
        
        x = p.x[:,κ]
        w = fill(-1*log(N),N)

        return (Particles(x,w),ess)
    else
        return (p,ess)
    end
end

# eventually write an object that allows for x0::Float64 if and only
# if kernel::UnivariateDistribution instead of this mess
struct Proposal
    x0::Union{Float64,Vector{Float64}}
    kernel::Function
end

function Proposal(model::GaussianSSM,x0::Union{Float64,Vector{Float64}})
    if isa(x0,Vector)
        kernel(xt) = MvNormal(model.A*xt,sqrt(model.Q))
    else
        kernel(xt) = Normal(model.A*xt,sqrt(model.Q))
    end

    Proposal(x0,kernel)
end

# note that Y == Matrix(length(yₜ),T)
function bootstrapFilter(
        Y::Matrix{Float64},
        N::Int64,
        prior::StateSpaceModel,
        prop::proposal,
        B::Float64=0.5
    )::Tuple{ParticleSet,Vector{Float}}

    T = size(Y,1)

    # initialize the algorithm
    ps  = ParticleSet(N,prior.dim_x,T)
    ess = zeros(Float64,T)

    Particles([prop.x0])
end