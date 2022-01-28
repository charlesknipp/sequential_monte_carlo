export randTruncatedMvNormal,logpdfTruncatedMvNormal,ProgressBar

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
        y::Vector{Float64},
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

        p1 = logpdf(Normal(condμ,sqrt(condΣ)),y[i])
        p2 = log(cdf(Normal(),u[i])-cdf(Normal(),l[i]))
        logprob += (p1-p2)

        x[i] = y[i]
    end

    return logprob
end

# write a struct to define a progress bar object type
using Printf

mutable struct ProgressBar
    # wrapped iterable
    wrapped::Any

    # bar structure
    width::Int64
    current::Int64

    # time variables
    start_time::UInt
    elapsed_time::UInt

    # misc variables
    printing_delay::UInt

    function ProgressBar(wrapped,width::Int64=20,delay::Number=.05)
        time  = time_ns()
        delay = trunc(UInt,delay*1e9)

        new(wrapped,width,0,time,time,delay)
    end
end

function update(pb::ProgressBar)
    # the bar should resemble this: [######    ] 10 seconds
    n_cells  = trunc(Int,pb.current*(pb.width/length(pb.wrapped)))
    progress = repeat("#",n_cells)
    space    = repeat(" ",abs(pb.width-n_cells))

    # format time elapsed
    time_elapsed = (pb.elapsed_time-pb.start_time)*1e-9
    m,s = divrem(round(Int,time_elapsed),60)
    time_elapsed = @sprintf("%02d:%02d",m,s)

    print("\r")
    print("[",progress,space,"] ",time_elapsed)
end


# change it's behavior in for loops as to not mess up the loop
function Base.iterate(pb::ProgressBar)
    pb.start_time = time_ns() - pb.printing_delay
    pb.current = 0

    update(pb)
    return iterate(pb.wrapped)
end

function Base.iterate(pb::ProgressBar,state)
    pb.elapsed_time = time_ns()
    pb.current += 1
    state = iterate(pb.wrapped,state)

    update(pb)
    return state
end