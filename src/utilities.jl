export randTruncatedMvNormal,logpdfTruncatedMvNormal,ProgressBar,bisection,tosvec,log_likelihood_fun

function sum_all_but(x,i)
    # not sure why this is...
    x[i] -= 1
    xsum = sum(x)
    x[i] += 1

    return xsum
end

tosvec(x) = reinterpret(SVector{length(x[1]),Float64},reduce(hcat,x))[:] |> copy

#=
    Create a function that calculates the log likelihood given θ to represent
    the model parameters. The output should be a function that accetps a vector
    θ::Vector and outputs the log likelihood for SMC²

    returns function : θ → p(y|θ)p(θ)

=#
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

function secant(
        f::Function,x0::Float64,x1::Float64;
        tol::AbstractFloat=1.e-12,maxiter::Int64=50
    )

    for _ in 1:maxiter
        y1 = f(x1)
        y0 = f(x0)
        x  = x1-y1*(x1-x0)/(y1-y0)

        if abs(x-x1) < tol
            return x
        end

        x0 = x1
        x1 = x
    end

    # throw an error upon exceeding max_iters
    error("Max iteration exceeded")
end

function bisection(
        f::Function,lower_bound::Number,upper_bound::Number;
        tol::AbstractFloat=1e-12,maxiter::Integer=50
    )

    f_lower_bound = f(lower_bound)
    # f_lower_bound*f(upper_bound) <= 0 || error("No real root")

    i = 0
    midpoint = upper_bound

    while upper_bound-lower_bound > tol
        i += 1
        i != maxiter || error("Max iteration exceeded")

        midpoint   = (lower_bound+upper_bound)/2
        f_midpoint = f(midpoint)

        if f_midpoint == 0
            break
        elseif f_lower_bound*f_midpoint > 0
            # Root is in the right half of [a,b]
            lower_bound   = midpoint
            f_lower_bound = f_midpoint
        else
            # Root is in the left half of [a,b]
            upper_bound = midpoint
        end
    end

    # this returns a value that will likely encourage resampling
    return upper_bound
end


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