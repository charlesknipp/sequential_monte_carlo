export StateSpaceModel,LinearGaussian,simulate

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