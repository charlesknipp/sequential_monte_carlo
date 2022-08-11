export Particles,reweight!,resample!,particle_type

#=
Particles{T} is a collector which defines a particle set containing Float64s or
Vector{Float64}. Particles are technically immutable but setting its properties
to Base.RefValue{Any} allows a user to modify that attribute as a 0 dimensional
array.

In order to manipulate this object we define the following methods:
    length(x::Particles)
    reweight(x::Particles,logw::Vector{Float64})
    resample(x::Particles)
=#

struct Particles{T<:Union{Float64,Vector{Float64}}}
    x::Vector{T}
    a::Vector{Int64}
    t::Base.RefValue{Int64}

    w::Vector{Float64}
    logμ::Base.RefValue{Float64}
    ess::Base.RefValue{Float64}

    function Particles{T}(x::Vector{T}) where T <: Union{Float64,Vector{Float64}}
        N = length(x)

        return new(
            x,
            collect(1:N),
            Ref(1),
            fill(1/N,N),
            Ref(-log(N)),
            Ref(1.0)
        )
    end
end

Base.length(X::Particles) = length(X.w)

function reweight!(X::Particles,logw::Vector{Float64})
    maxw = maximum(logw)
    w    = exp.(logw.-maxw)
    sumw = sum(w)

    # change the "immutable" struct by RefValue manipulation
    X.logμ[] = maxw + log(sumw) - log(length(logw))
    X.w[:]  .= w/sumw
    X.ess[]  = 1.0/sum(X.w.^2)

    return X.logμ[]
end

function resample!(X::Particles)
    X.a[:] .= wsample(1:length(X),X.w,length(X))
end

particle_type(X::Particles{T}) where {T} = T
