include(joinpath(pwd(),"src/sequential_monte_carlo.jl"))

using .SequentialMonteCarlo
using Plots,Distributions,Random

# model definition
params = LinearGaussian(0.8,1.0,1.0,1.0)
model  = StateSpaceModel(params)

# simulation
# Random.seed!(123)
x,y = simulate(model,100)


# kalman filter plot
xkf = kalmanFilter(y,params)
	
lqkf = xkf[:,2] - xkf[:,1]
uqkf = xkf[:,3] - xkf[:,2]

kfplot = plot(
    xkf[:,2],rib=(lqkf,uqkf),lab="kalman filter",
    fa=0.2,lw=1.5,lc=:red,fc=:red
)

plot!(kfplot,x,lab="simulated x",lw=1.5,lc=:black,ls=:dash)

kfplot = plot(kfplot;size=default(:size).*(1.25,.75),grid=false)

savefig(kfplot,"visuals/plots/kfplot.png")


# bootstrap filter plot
# Random.seed!(123)
xbf_p = bootstrapFilter(1000,y,model,Inf)
xbf = zeros(Float64,100,3)

for t in 1:100
    xbf[t,:] = quantile(xbf_p.p[t].x,[.25,.50,.75])
end

lqbf = xbf[:,2] - xbf[:,1]
uqbf = xbf[:,3] - xbf[:,2]

bfplot = plot(
    xbf[:,2],rib=(lqbf,uqbf),lab="bootstrap filter",
    fa=0.2,lw=1.5,lc=:green,fc=:green
)

plot!(bfplot,x,lab="simulated x",lw=1.5,lc=:black,ls=:dash)

bfplot = plot(bfplot;size=default(:size).*(1.25,.75),grid=false)

savefig(bfplot,"visuals/plots/bfplot.png")

#=
# auxiliary particle filter plot
xpf = auxiliaryParticleFilter(1000,y,sims)[3]
	
lqpf = xpf[:,2] - xpf[:,1]
uqpf = xpf[:,3] - xpf[:,2]

pfplot = plot(
    xpf[:,2],rib=(lqpf,uqpf),lab="auxiliary particle filter",
    fa=0.2,lw=1.5,lc=:blue,fc=:blue
)

plot!(pfplot,x,lab="simulated x",lw=1.5,lc=:black,ls=:dash)

pfplot = plot(pfplot;size=default(:size).*(1.25,.75),grid=false)

savefig(pfplot,"visuals/plots/apfplot.png")
=#

# filter summary plot
cumplot = plot(xkf[:,2],lab="kalman filter",lc=:red,la=.5,lw=2)
plot!(cumplot,xbf[:,2],lab="bootstrap filter",lc=:green,la=.5,lw=2)
#plot!(cumplot,xpf[:,2],lab="auxiliary particle filter",lc=:blue,la=.5,lw=2)

cumplot = plot(cumplot;size=default(:size).*(1.25,.75),grid=false)

savefig(cumplot,"visuals/plots/cumplot.png")