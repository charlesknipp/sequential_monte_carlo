using LinearAlgebra
using Distributions

############################### MiscFunctions.jl ###############################

function mean(vec::Vector{T}) where T <: Number
    # calculate the arithmetic mean of a vector of real numbers
    μ = sum(vec)/length(vec)
    return μ
end

# for the mean over a matrix
function mean(mat::Matrix{T},dim::Int64=1) where T <: Number
    μ = sum(mat,dims=dim)/size(mat,dim)
    return μ
end


# generates N random points of a truncated normal distribution
function randTruncatedMvNormal(
        N::Int64,
        μ::Vector{Float64},
        Σ::Matrix{Float64},
        l::Vector{T},
        u::Vector{T}
    ) where T <: Number

    k = length(μ)
    x = ones(Float64,k,N).*μ

    # perform this N times over; either vectorize or iterate over another loop
    for i in 1:k
        # convert covariance into a block matrix
        Σ11,Σ22 = Σ[i,i],Σ[1:end .!= i,1:end .!= i]
        Σ12,Σ21 = Σ[1:end .!= i,i]',Σ[1:end .!= i,i]

        x2 = x[1:end .!= i,1]
        μ1,μ2 = μ[i],μ[1:end .!= i]

        condΣ = Σ11 - Σ12*inv(Σ22)*Σ21
        condμ = μ1 + Σ12*inv(Σ22)*(x2-μ2)

        lcdf = cdf(Normal(condμ,sqrt(condΣ)),l[i])
        ucdf = cdf(Normal(condμ,sqrt(condΣ)),u[i])
        prob = rand(Uniform(lcdf,ucdf),N)

        x[i,:] = quantile(Normal(),prob)*sqrt(condΣ) .+ condμ
    end

    return x
end


# Calculates the log density at θ given parameters μ and Σ
function logpdfTruncatedMvNormal(
        proposal::Vector{Float64},
        μ::Vector{Float64},
        Σ::Matrix{Float64},
        l::Vector{T},
        u::Vector{T}
    ) where T <: Number

    k = length(μ)
    x = μ

    logprob = 0

    for i in 1:k
        # convert covariance into a block matrix
        Σ11,Σ22 = Σ[i,i],Σ[1:end .!= i,1:end .!= i]
        Σ12,Σ21 = Σ[1:end .!= i,i]',Σ[1:end .!= i,i]

        x2 = x[1:end .!= i]
        μ1,μ2 = μ[i],μ[1:end .!= i]

        condΣ = Σ11 - Σ12*inv(Σ22)*Σ21
        condμ = μ1 + Σ12*inv(Σ22)*(x2-μ2)

        p1 = logpdf(Normal(condμ,sqrt(condΣ)),proposal[i])
        p2 = log(cdf(Normal(),u[i])-cdf(Normal(),l[i]))
        logprob += (p1-p2)

        x[i] = proposal[i]
    end

    return logprob
end

############################## StateSpaceModel.jl ##############################

abstract type AbstractSSM end
abstract type ModelParameters end

struct StateSpaceModel <: AbstractSSM
    # define general structure using functions
    transition::Function
    observation::Function
    
    # need the dims for initializing states...maybe...
    dim_x::Int64
    dim_y::Int64
end


# for now assume x0 and Σ0 are known
struct LinearGaussian <: ModelParameters

    A::Union{Float64,Matrix{Float64}}
    B::Union{Float64,Matrix{Float64}}

    Q::Union{Float64,Matrix{Float64}}
    R::Union{Float64,Matrix{Float64}}

    # implicitly defined by the constructor
    dim_x::Int64
    dim_y::Int64

    function LinearGaussian(A,B,Q,R)
        # determine the dimensions
        dim_x,dim_y = size(A,1),size(B,1)

        @assert dim_x == size(A,2) "A is not a square matrix"
        @assert dim_y == size(B,2) "B is not a square matrix"

        @assert size(Q,1) == size(Q,2) "Q is not a square matrix"
        @assert size(R,1) == size(R,2) "R is not a square matrix"

        @assert dim_x == size(Q,1) "A,Q dimension mismatch"
        @assert dim_y == size(R,1) "B,R dimension mismatch"

        @assert issymmetric(Q) "Q is not symmetric"
        @assert issymmetric(R) "R is not symmetric"

        @assert isposdef(Q) "Q is not positive definite"
        @assert isposdef(R) "R is not positive definite"

        # construct the new object
        new(A,B,Q,R,dim_x,dim_y)
    end
end


function StateSpaceModel(params::LinearGaussian)
    # import parameters
    A,B = params.A,params.B
    Q,R = params.Q,params.R

    dim_x = params.dim_x
    dim_y = params.dim_y

    # depending on the type of input set the kernel
    Kx = (dim_x == 1) ? Normal : MvNormal
    Ky = (dim_y == 1) ? Normal : MvNormal

    f(xt) = Kx(A*xt,sqrt(Q))
    g(xt) = Ky(B*xt,sqrt(R))

    return StateSpaceModel(f,g,dim_x,dim_y)
end


function simulate(model::StateSpaceModel,T::Int64)
    y = (model.dim_y == 1) ? 0.0 : zeros(Float64,model.dim_y)
    y = fill(y,T)
    
    x = (model.dim_x == 1) ? 0.0 : zeros(Float64,model.dim_x)
    x = fill(x,T)

    for t in 1:(T-1)
        y[t] = rand(model.observation(x[t]))
        x[t+1] = rand(model.transition(x[t]))
    end

    y[T] = rand(model.observation(x[T]))

    return (x,y)
end

################################# Particles.jl #################################

# try to make the Particles type immutable, which may entail getting rid of the
# ParticleSet class

const ParticleType = Union{Float64,Vector{Float64}}

mutable struct Particles{T<:ParticleType}
    # necessary parameters
    x::Vector{T}
    logw::Vector{Float64}

    # other parameters, may be inefficient to calculate them at every call
    w::Vector{Float64}
    ess::Float64
    logμ::Float64

    function Particles(x::Vector{T},logw::Vector{Float64}) where T<:ParticleType
        # account for underflows in weights more thoroughly such that if logw
        # === NaN then set logw = -Inf
        maxw = maximum(logw)
        w    = exp.(logw.-maxw)
        sumw = sum(w)

        logμ = maxw + log(sumw/length(logw))
        w    = w/sumw
        ess  = 1.0/sum(w.^2)
        
        new{T}(x,logw,w,ess,logμ)
    end
end

function Particles(particles::Vector{T}) where {T<:ParticleType}
    N_particles = length(particles)
    log_weights = fill(-1*log(N_particles),N_particles)

    return Particles(particles,log_weights)
end

function Particles(N::Int64,dims::Int64=1)
    particles = (dims == 1) ? fill(0.0,N) : fill(zeros(Float64,dims),N)
    log_weights = fill(-1*log(N),N)

    return Particles(particles,log_weights)
end

function reweight(particles::Particles,log_weights::Vector{Float64})
    particles = particles.x

    # this returns a new object, which makes mutability of this struct obsolete
    return Particles(particles,log_weights)
end

#function mean(p::Particles)
#    # convert the vector of vectors to a matrix
#    k = length(p.x[1])
#    x = reduce(hcat,p.x)
#    max_x = maximum(x,dims=2)
#    w = exp.(p.logw.-maximum(p.logw))
#
#    μ = [exp(max_x[i])*sum(w.*exp.(x[i,:].-max_x[i]))/sum(w) for i in 1:k]
#    
#    return μ
#end


mutable struct ParticleSet{T<:ParticleType}
    p::Vector{Particles{T}}
end

function ParticleSet(N::Int64,dim::Int64,T::Int64)
    particle_set = [Particles(N,dim) for _ in 1:T]

    return ParticleSet(particle_set)
end

#function mean(ps::ParticleSet)
#    return [mean(p) for p in ps.p]
#end


function ESS(logx::Vector{Float64})
    max_logx = maximum(logx)
    x_scaled = exp.(logx.-max_logx)

    ESS = sum(x_scaled)^2 / sum(x_scaled.^2)
    if ESS == NaN; ESS = 0.0 end

    return ESS
end

function resample(p::Particles,B::Number=Inf)
    N = length(p.x)

    if p.ess < B*N
        a = wsample(1:N,p.w,N)
        x = p.x[a]

        return Particles(x)
    else
        return p
    end
end

############################## ParticleFilters.jl ##############################

# this works, but could honestly perform a little better
function bootstrapFilter(
        N::Int64,
        y::Vector{Float64},
        prior::StateSpaceModel,
        B::Float64 = 0.5,
        proposal = nothing
    )
    T = length(y)

    if proposal === nothing
        proposal = prior
    end

    # initialize algorithm
    ps  = ParticleSet(N,prior.dim_x,T)

    # in the case where x0 is located at the origin
    x0 = (prior.dim_x == 1) ? 0.0 : zeros(Float64,prior.dim_x)
    p0 = Particles(rand(proposal.transition(x0),N))

    ps.p[1] = resample(p0,B)

    for t in 2:T
        xt = rand.(proposal.transition.(ps.p[t-1].x))
        wt = logpdf.(prior.observation.(xt),y[t])

        # simplify calculations if the proposal is not provided
        if !(proposal === nothing)
            wt += logpdf.(prior.transition.(ps.p[t-1].x),xt)
            wt -= logpdf.(proposal.transition.(ps.p[t-1].x),xt)
        end

        ## for higher dimensions of x
        # xt = [xt[:,i] for i in 1:size(xt,2)]
        wt += ps.p[t-1].logw
        ps.p[t] = resample(Particles(xt,wt),B)
    end

    return ps
end

###################################  SMC2.jl ###################################

# not sure this works, so I might need some testing
function randomWalk(θ::Particles,c::Float64=0.5)
    k = length(θ.x[1])
    x = reduce(hcat,θ.x)

    # calculate the weighted mean
    μ = [(sum(θ.w .* x[i,:]))/sum(θ.w) for i in 1:k]
    
    # calculate the weighted covariance
    adjx = [x[:,m] - μ for m in 1:length(θ.x)]
    Σ = sum(θ.w[m]*adjx[m]*adjx[m]' for m in 1:M)/sum(w)
    # Σ = StatsBase.cov(x,μ,θ.w)

    return rand(MvNormal(μ,c*Σ))
end

# not complete, and also not necessary YET
function bootstrapStep(
        t::Int64,
        Θ::Vector{StateSpaceModel},
        N::Int64,
        y::Vector{Float64},
        Xt::Vector{Particles},
        B::Float64 = 0.5
    )
    M = length(Xt)

    for m in 1:M
        xt = rand.(Θ[m].transition.(Xt[m].x),N)
        wt = logpdf.(Θ[m].observation.(xt),y[t])
        Xt[m] = resample(Particles(xt,wt),B)
    end

    return Xt
end

# I should write a method to perform a filter for t iterations
function SMC2(
        N::Int64,
        M::Int64,
        y::Vector{Float64},
        θ0::Vector{Float64},
        prior::Function,
        B::Float64 = 0.5,
        model = LinearGaussian
    )

    T = length(y)
    k = length(θ0)

    θ = prior(M,θ0,Matrix{Float64}(I,k,k))
    θ = Particles([θ[:,m] for m in 1:M])

    Θ = [StateSpaceModel(model(θ.p[m].x...)) for m in 1:M]

    x0 = (Θ[1].dim_x == 1) ? 0.0 : zeros(Float64,Θ[1].dim_x)
    Xt = [Particles(rand(Θ[m].transition(x0),N)) for m in 1:M]

    # perform iteration t of the bootstrap filter and reweight θ particles
    for t in 1:T
        Xt = bootstrapStep(t,Θ,N,y,Xt,B)
        θ  = reweight(θ,θ.logw+[Xtm.logμ for Xtm in Xt])
        
        # perform MH steps in case of degeneracy
        if θ.ess < B*N
            newθ = randomWalk(θ,0.5)
            newΘ = [StateSpaceModel(model(newθ.p[m].x...)) for m in 1:M]

            # perform another PF from 1:t
            newx0 = (newΘ[1].dim_x == 1) ? 0.0 : zeros(Float64,newΘ[1].dim_x)
            newXt = [Particles(rand(newΘ[m].transition(newx0),N)) for m in 1:M]

            for k in 1:t
                newXt = bootstrapStep(k,newΘ,N,y,newXt,B)
                newθ  = reweight(newθ,newθ.logw+[newXt[m].logμ for m in 1:M])
            end

            # Z(θ) ≡ likelihood
            Zt    = Xt.logμ
            newZt = newXt.logμ

            # p(θ) ≡ pdf of prior
            pt    = logpdf(prior,θ)
            newpt = logpdf(prior,newθ)

            # T(θ) ≡ pdf of proposal distribution
            #Tt    = logpdf()
            #newTt = logpdf()
        end
    end
end

############################## DensityTempered.jl ##############################

# this one needs quite a bit of work, but that's okay since I already have a
# working version of the old code

function densityTemperedSMC(
        N::Int64,
        M::Int64,
        y::Vector{Float64},
        θ0::Vector{Float64},
        model=LinearGaussian
    )
    T = length(y)
    k = length(θ0)

    lb = [-1.0,-1.0,0.0,0.0]
    ub = [1.0,1.0,2.0,2.0]
    Σ0 = Matrix{Float64}(I,k,k)

    θ = randTruncatedMvNormal(N,θ0,Σ0,lb,ub)
    θ = Particles([θ[:,i] for i in 1:N],fill(-1*log(N),N))

    ## try catch this later on to make sure the model is well defined
    # model = eval(Meta.parse(model))

    for i in 1:N
        # see equation (4) from Duan & Fulop
        mod_i = StateSpaceModel(model(θ.x...))
        
        # find the likelihood p_hat as Duan & Fulop call it

        # weights evenly for l=1
        S[1][i] = -log(N)
    end
end

################################## Testing.jl ##################################

using Printf

test_params = LinearGaussian(1.0,1.0,1.0,1.0)
test_model  = StateSpaceModel(test_params)

x,y  = simulate(test_model,100)
xs,_ = bootstrapFilter(1000,y,test_model)

for t in 1:100
    μ = sum(xs.p[t].x)/1000
    println(@sprintf("x_pf: %.5f\tx_sim: %.5f",μ,x[t]))
end
