include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Printf,Random,Distributions

test_params = LinearGaussian(1.0,1.0,1.0,1.0)
test_model  = StateSpaceModel(test_params)

# initializing needs some work
Random.seed!(1234)
x,y = simulate(test_model,100)

Random.seed!(1234)
xs_bf,logZ_bf = bootstrapFilter(1000,y,test_model,Inf)
xs_kf,logZ_kf = kalmanFilter(y,test_params)

for t in 1:100
    println(@sprintf("logZ_bf: %.5f\tlogZ_kf: %.5f",logZ_bf[t],logZ_kf[t]))
end


using BenchmarkTools

# rewritten bootstrap method, slightly faster (emphasis on slightly)
function bf(N::Int64,y::Vector{Float64},prior::StateSpaceModel)
    T = length(y)

    # initialize algorithm
    logZ = zeros(Float64,T)

    # in the case where x0 is located at the origin
    x0 = (prior.dim_x == 1) ? 0.0 : zeros(Float64,prior.dim_x)
    p0 = Particles(rand(prior.transition(x0),N))

    logZ[1] = p0.logμ
    ps = resample(p0)

    for t in 2:T
        xt = rand.(prior.transition.(ps.x))
        wt = logpdf.(prior.observation.(xt),y[t])
        pt = Particles(xt,wt)

        logZ[t] = pt.logμ
        ps = resample(pt)
    end

    return logZ
end

Random.seed!(1234)
logZ_bf1 = bf(1000,y,test_model)

println()
for t in 1:100
    println(@sprintf("logZ_bf: %.5f\tlogZ_bf1: %.5f",logZ_bf[t],logZ_bf1[t]))
end

function timebf1()
    Random.seed!(123)
    _,logZ = bootstrapFilter(1000,y,test_model,Inf)
    return logZ
end

function timebf2()
    Random.seed!(123)
    logZ = bf(1000,y,test_model)
    return logZ
end

@benchmark timebf1()
@benchmark timebf2()