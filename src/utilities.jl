export randTruncatedMvNormal,logpdfTruncatedMvNormal,ProgressBar,secant

using Printf

function secant(
        f::Function,x0::Number,x1::Number;
        tol::AbstractFloat=1e-5,maxiter::Int64=50
    )

    for _ in 1:maxiter
        y1 = f(x1)
        y0 = f(x0)
        x  = (x1-y1)*(x1-x0)/(y1-y0)

        if abs(x-x1) < tol
            return x
        end

        x0 = x1
        x1 = x
    end

    # throw an error upon exceeding max_iters
    error("Max iteration exceeded")
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