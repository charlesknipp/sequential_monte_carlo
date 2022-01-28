export Particles,reweight,ParticleSet,ESS,resample

# try to make the Particles type immutable, which may entail getting rid of the
# ParticleSet class

const ParticleType = Union{Float64,Vector{Float64},Vector{Matrix{Float64}}}

mutable struct Particles{T<:ParticleType}
    # necessary parameters
    x::Vector{T}
    logw::Vector{Float64}

    # other parameters, may be inefficient to calculate them at every call
    w::Vector{Float64}
    ess::Float64
    logμ::Float64

    function Particles(x::Vector{T},logw::Vector{Float64}) where T <: ParticleType
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

Base.length(particles::Particles) = length(particles.logw)

function reweight(particles::Particles,log_weights::Vector{Float64})
    particles = particles.x

    # this returns a new object, which makes mutability of this struct obsolete
    return Particles(particles,log_weights)
end


mutable struct ParticleSet{T<:ParticleType}
    p::Vector{Particles{T}}
end

function ParticleSet(N::Int64,dim::Int64,T::Int64)
    particle_set = [Particles(N,dim) for _ in 1:T]

    return ParticleSet(particle_set)
end

Base.length(ps::ParticleSet) = length(ps.p)

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