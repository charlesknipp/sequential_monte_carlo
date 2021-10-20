using Plots

include("dynamic_model.jl")
include("particle_filter.jl")

# sim
sims = NDLM(0.8,1.0,1.0,1.0)
x,y = simulate(100,sims)


# kalman filter
xkf = kalmanFilter(y,sims)
	
lqkf = xkf[:,2] - xkf[:,1]
uqkf = xkf[:,3] - xkf[:,2]

kfplot = plot(
    xkf[:,2],rib=(lqkf,uqkf),lab="kalman filter",
    fa=0.2,lw=1.5,lc=:red,fc=:red
)

plot!(kfplot,x,lab="simulated x",lw=1.5,lc=:black,ls=:dash)

plot(kfplot;size=default(:size).*(1.25,.75),grid=false)


# bootstrap filter
xbf = bootstrapFilter(1000,y,sims)[3]
	
lqbf = xbf[:,2] - xbf[:,1]
uqbf = xbf[:,3] - xbf[:,2]

bfplot = plot(
    xbf[:,2],rib=(lqbf,uqbf),lab="bootstrap filter",
    fa=0.2,lw=1.5,lc=:green,fc=:green
)

plot!(bfplot,x,lab="simulated x",lw=1.5,lc=:black,ls=:dash)

plot(bfplot;size=default(:size).*(1.25,.75),grid=false)


# auxiliary particle filter
xpf = auxiliaryParticleFilter(1000,y,sims)[3]
	
lqpf = xpf[:,2] - xpf[:,1]
uqpf = xpf[:,3] - xpf[:,2]

pfplot = plot(
    xpf[:,2],rib=(lqpf,uqpf),lab="auxiliary particle filter",
    fa=0.2,lw=1.5,lc=:blue,fc=:blue
)

plot!(pfplot,x,lab="simulated x",lw=1.5,lc=:black,ls=:dash)

plot(pfplot;size=default(:size).*(1.25,.75),grid=false)

cumplot = plot(xkf[:,2],lab="kalman filter",lc=:red,la=.5,lw=2)
plot!(cumplot,xbf[:,2],lab="bootstrap filter",lc=:green,la=.5,lw=2)
plot!(cumplot,xpf[:,2],lab="auxiliary particle filter",lc=:blue,la=.5,lw=2)

plot(cumplot;size=default(:size).*(1.25,.75),grid=false)