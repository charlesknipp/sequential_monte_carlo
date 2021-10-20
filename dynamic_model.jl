using LinearAlgebra,Distributions,Statistics


"""
    NDLM(A,B,Q,R,x0=0.0,Σ0=1.0)

Defines a class of Gaussian dynamic linear models by its parameters.
The structure of these models is observed below:

* y[t]   = B*x[t] + δ   s.t. δ ~ N(0,R)
* x[t+1] = A*x[t] + ε   s.t. ε ~ N(0,Q)
* x[1] ~ N(x0,Σ0)

"""
struct NDLM
    # model coefficitents
    A::Float64
    B::Float64

    # error variances
    Q::Float64
    R::Float64

    # initial parameters
    x0::Float64
    Σ0::Float64

    # overwrite the default constructor to allow for default args
    NDLM(A,B,Q,R,x0=0.0,Σ0=1.0) = new(A,B,Q,R,x0,Σ0)
end


"""
    simulateNDLM(T,x0,Σ0,A,B,Q,R)

Simulate a normal dynamic linear model using the following form...

* y[t]   = B*x[t] + δ   s.t. δ ~ N(0,R)
* x[t+1] = A*x[t] + ε   s.t. ε ~ N(0,Q)
* x[1] ~ N(x0,Σ0)

The purpose of this function is to make subset testing a little easier.
It will not be used in any of the following algorithms, but will test
whether the results produced are consistent with simulations.

"""
function simulate(T::Int64,model::NDLM)
    x = ones(Float64,T)*rand(Normal(model.x0,model.Σ0))
    y = zeros(Float64,T)

    for t in 2:T
        x[t] = (model.A)*x[t-1] + rand(Normal(0,sqrt(model.Q)))
        y[t] = (model.B)*x[t] + rand(Normal(0,sqrt(model.R)))
    end

    return (x,y)
end