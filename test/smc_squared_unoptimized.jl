using LinearAlgebra,Distributions,Printf

#=
    This script performs parameter estimation with smc2 without using parallel computing
    (Code can be easily modified to do each individual run of smc2 in parallel)

    This code implements smc2 with a state-process that is one-dimensional.
    However, with slight modifications, multidimensional state-process can be
    used as well

    If one wants to use this script with another model, atleast functions
    "X_transition" ans "Obs_density must be modified along with small details
    in "PF_call" and "PMHkernel" that relate to number of parameters, initial
    X and priors.

    This code was made for Master's thesis: "Sequential estimation of
    state-space model parameters" by Joona Paronen
    see the thesis for references
=#

function X_transition(Xold,θ,Nx)
    # this function is model specific
    # Markov transition of the model f(x_{t} | x_{t-1})
    U = randn(1,Nx)
    Xnew = θ[1] + θ[2].*(Xold-θ[1]) + θ[3].*U

    return Xnew,U
end


function Obs_density(Ypoint,X,U,θ)
    # this function is model specific
    # G( y_t | x_t ) ~ N(beta*x_t, exp(x_t/2))
    
    w = Ypoint - θ(4)*X - exp(X/2).*U*θ(5)
    σ = exp(X/2)*sqrt(1-θ(5)^2)
    return log(1.0/((2*pi)^(0.5).*σ).*exp(-0.5*(w).^2.0/σ.^2))
end

## simulate data

function simulateData(params,numObs)
    #####
    μ = params.μ
    ρ = params.ρ
    σ = params.σ
    β = params.β
    φ = params.φ
    #####

    # randomness
    u = randn()
    v = φ*u+sqrt(1-φ^2)*randn()

    # initialize
    x[1] = μ+sqrt(σ^2/(1-ρ^2))*u
    y[1] = β*x(1)+exp(x(1)/2)*v

    for t in 2:numObs
        U = randn()
        V = φ*U + sqrt(1-φ^2)*randn()

        x[t] = μ + ρ*(x[t-1]-μ) + σ*U
        y[t] = β*x[t] + exp(x[t]/2)*V
    end

    return x,y
end



function PF_call(Yt,theta,Nx,add=0)

    # BOOTSTRAP PARTICLE FILTER
    # this function computes p_bar (y_{1:t}) for each individual theta-particle
    #
    # INPUT:
    #           Yt - observed data until time t
    #        theta - parameters (5x1 vector in this case)
    #           Nx - number of X-particles
    #          add - 0 or 1, 1 if particle addition happens
    #        
    # OUTPUT:
    #         newW - likelihood of proposal theta
    #    newXparts - propagated X-particles at time T
    #  newXweights - weigths of those particles at time T
    #         flag - error flag in case weigths are really close to zero

    T = length(Yt)
    X_weights = zeros(T,Nx)     # weigths
    Xparticles = zeros(T,Nx)    # particles

    # first state
    Xparticles[1,:] = theta[1]+sqrt(theta[3]^2/(1-theta[2]^2))*randn(1,Nx)
    flag = 1
    u = randn(1,Nx)

    for t = 1:T
        if t == 1
            X_weights[t,:] = Obs_density(Yt[t],Xparticles[t,:],u,theta) # save the weights
        else
            a = wsample(weight_mod,Nx)  # resample every time
            Xparticles[t,:],u = X_transition(Xparticles[t-1,a],theta,Nx)    # propagate forward
            X_weights[t,:] = Obs_density(Yt[t],Xparticles[t,:],u,theta)     # save the weigths
        end

        weight_mod = exp.(X_weights[t,:].-maximum(X_weights[t,:]))

        if sum(weight_mod) < 1e-15
            # if weigths "carry no meaning" == really small, discard
            # them and this proposal theta
            flag = 0
            break
        end
        weight_mod = weight_mod/sum(weight_mod) # normalize
    end

    if flag == 0
        newZ=[]
        newXparts=[]
        newXweights=[]
        return
    end

    if add
        newXparts = Xparticles      # save the last x-particles
        newXweights = X_weights     # and their weigths

        # compute the p(y_{1_t}): sum all the log-likelihood increments
        newZ = cumsum(log(1/Nx*sum(exp(X_weights),2)))
    else
        newXparts = Xparticles[T,:]     # save the last x-particles
        newXweights = X_weights[T,:]    # and their weigths

        # compute the p(y_{1_t}): sum all the log-likelihood increments
        newZ = sum(log(1/Nx*sum(exp(X_weights),2)))
    end
    
    return newZ,newXparts,newXweights,flag
end

function PMHkernel(oldZ,oldstate,oldX,oldXweights,ydata,Nsteps,R,N,Priors,acc)

    npar = length(oldstate)

    for moves in 1:Nsteps 
        proposal = oldstate + R*randn(npar,1)

        Z_proposal,X_proposal,Xweights_proposal,errorflag = PF_call(ydata,proposal,N,0)
        #########################
        if errorflag == 0 # saves time, look explanation inside PF_call
            acceptance_ratio = -Inf
        else
            acceptance_ratio = (Z_proposal-oldZ)
            acceptance_ratio += logpdf(Priors.mu,proposal(1))-logpdf(Priors.mu,oldstate(1))
            acceptance_ratio += logpdf(Priors.rho,proposal(2))-logpdf(Priors.rho,oldstate(2))
            acceptance_ratio += logpdf(Priors.sigma,proposal(3))-logpdf(Priors.sigma,oldstate(3))
            acceptance_ratio += logpdf(Priors.beta,proposal(4))-logpdf(Priors.beta,oldstate(4))
            acceptance_ratio += logpdf(Priors.phi,proposal(5))-logpdf(Priors.phi,oldstate(5))
        end

        if log(rand()) <= min(1,acceptance_ratio) && isfinite(acceptance_ratio)
            oldstate = proposal       
            # now since we accepted the new theta^{m}
            # its cumulative log-weigth changes as well.
            oldZ = Z_proposal      
            # set the new X-particles (for particular theta^{m}) and their weigths because
            # the move was accepted so the states must move, too.
            oldX = X_proposal
            oldXweights = Xweights_proposal
            acc+=1
        end
    end

    newZ = oldZ
    newstate = oldstate
    newX = oldX
    newXweights = oldXweights

    return newstate,newZ,newX,newXweights,acc
end

#=
    example simple stochastic volatility model: 
    -------------------------------------------------------
    x_0 ~ N(μ, σ^2/(1-ρ^2))
    x_t = μ + ρ*(x_{t-1}-μ) + σ*U_t      U ~ N(0,1)
    y_t|x_t = β*x_t + exp(x_t/2)*V_t
    corr(U,V) = φ
    <<<<
    parameter vector: θ = [μ,ρ,σ,β,φ]
=#

## define prior densities:


struct Priors
    μ
    ρ
    σ
    β
    φ
end

Priors.μ = Normal(0.0,2.0)
Priors.ρ = Beta(9,1)
Priors.σ = Gamma(2,2)
Priors.β = Normal(0,1)
Priors.φ = Uniform(0,1)


params.μ = -1.0
params.ρ = 0.9
params.σ = 0.8
params.β = 0.0
params.φ = -0.5

# params.mu=random(Priors.mu);
# params.rho=random(Priors.rho);
# params.sigma=random(Priors.sigma);
# params.beta=random(Priors.beta);
# params.phi=random(Priors.phi);

T = 700 # number of observations
npar = length(params)  # number of model parameters
y,x = simulateData(params,T)

## MATLAB plotting shit
#  subplot(211);
#  plot(x,'Color','k');
#  title('State process X_t');
#  subplot(212)
#  plot(y.^2,'Color','k');
#  title('Squared observations Y_t');
#  xlabel('t');
#  y = y(:);
#  x = x(:);

## SMC² settings

## necessary inputs:
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

NT = 1000                       # number of parameter particles
init_X_particles = 100          # number of initial x-particles
Np_x_max = 1000                 # maximum # of x-particles (for allocation, can be as big as memory allows)
ESSfrac = 0.5                   # effective sample size = NT * ESSfrac, choose between 0 and 1
Nsteps = 3                      # how many rejuvenation steps
acc_threshold = 0.15            # guideline to tune covariance and add x-particles 
nruns = 5                       # how many runs of smc2

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# mcmc settings:
sd = 2.38^2/npar    # scaling factor from literature
e  = 1e-10          # to prevent singularity
I  = eye(npar)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>

T = length(y)   # number of observations

# sample initial theta-particles
thetaParticles = zeros(npar,NT,T+1,nruns)   # θ[5,Nx,T,n_runs]
fld = fieldnames(Priors)

for i in 1:npar
    # assign θ particels by taking a random sample of each distribtuion
    thetaParticles[i,:,1,:] = random(Priors.(fld{i}),1,NT,1,nruns)
end

# allocate variables that are saved
NX = zeros(T,nruns)
accept_rates = zeros(T,2,nruns)
Z = zeros(NT,T,nruns)
L = zeros(T,nruns)


## Run 'nruns' of smc2
for run in 1:nruns
    NX[:,run] = init_X_particles
    
    # These can be saved for each run if one is interested to do filtering/smoothing
    # or plotting ESS
    
    # ------------------------------
    Xweights = zeros(Float64,T,Np_x_max,NT)
    Xparticles = zeros(Float64,T,Np_x_max,NT)
    ParamWeights = zeros(Float64,NT,T)
    ESS = zeros(Float64,T)
    # ------------------------------

    # Particle system is a set of variables:
    # ( thetaParticles, Xparticles, Xweights, Paramweights )
    U = randn(1,NX[1,run])
    c,R = 1,1

    # go through every timestep until T
    for t in 1:T 
        # Move one step ahead with particle filter
        #######################################################################
        for m in 1:NT # for each theta-particle (m)
            if t == 1
                Xweights[t,1:NX[t,run],m] = Obs_density(y[t],Xparticles[t,1:NX[t,run],m],U,thetaParticles[:,m,t,run])
            else
                # practical trick to maintain the right order of weights
                # while making them larger (effect is eliminated in
                # normalization)
                temp = exp(Xweights[t-1,1:NX[t,run],m]-maximum(Xweights[t-1,1:NX[t,run],m]))
                temp = temp/sum(temp)
                ###########################
                a = wsample(temp,NX[t,run]) # resample

                # propagate through system model
                Xparticles[t,1:NX[t,run],m],U = X_transition(Xparticles[t-1,a,m],thetaParticles[:,m,t,run],NX[t,run])                
                Xweights[t,1:NX[t,run],m]     = Obs_density(y[t],Xparticles[t,1:NX[t,run],m],U,thetaParticles[:,m,t,run])
            end              
        end

        # Compute incremental likelihood and update marginal log-likelihood
        # and parameter weights
        
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
        # sum over x-particles to get weight of each theta-particle
        if t == 1
            ParamWeights[:,1] = squeeze(sum(1/NX[t,run]*exp(Xweights[t,1:NX(t,run),:]),2))
            Z[:,1,run] = log(ParamWeights[:,1])
            L[1,run] = sum(ParamWeights[:,1])
        else
            likelihood_increment = squeeze(sum(1/NX[t,run]*exp(Xweights[t,1:NX[t,run],:]),2))         
            ParamWeights[:,t]    = ParamWeights[:,t-1].*likelihood_increment
            
            # marginal likelihood is the product of incremental likelihood
            Z[:,t,run] = Z[:,t-1,run] + log(likelihood_increment)
            L[t,run]   = 1/sum(ParamWeights[:,t-1])*sum(ParamWeights[:,t-1].*likelihood_increment)
        end

        # normalize
        
        ParamWeights[:,t] = ParamWeights[:,t]./sum(ParamWeights[:,t])
        ESS[t] = 1/sum(ParamWeights[:,t].^2)

        do_we_move = 0      # dummy to indicate whether move is done at time t
        acc = 0             # accept counter
        addParticles = 0    # another dummy

        if ESS[t]/NT < ESSfrac || isnan(ESS[t]) # MATLAB: || ~isreal(ESS[t])
            do_we_move = 1 # yes, we do move
            # resample theta-weigths (most meaningful parameters weigths)
            A = wsample(ParamWeights[:,t],NT)
            
            # Set the resampled particle system 
            # ---------------------------------------------------
            thetaParticles[:,:,t,run] = thetaParticles[:,A,t,run]
            Z[:,t,run] = Z[A,t,run]

            Xparticles[t,1:NX[t,run],:] = Xparticles[t,1:NX[t,run],A]            
            Xweights[t,1:NX[t,run],:]   = Xweights[t,1:NX[t,run],A]
            # ---------------------------------------------------

            # Adaptation/tuning of proposal
            covariance = cov(thetaParticles[:,:,t,run]')
            if norm(covariance) < 1e-12
                warning("Covariance too low, consider stopping")
                R = 1e-2*I
                c = 1
            else
                R = chol(c*sd*covariance+e*I).L # cholesky decomposition (lower)
            end

            println("Rejuvenating particles")
            
            for m in 1:NT
                newpars,newZ,newXs,newXweights,acc = PMHkernel(
                    Z[m,t,run],
                    thetaParticles[:,m,t,run],
                    Xparticles[t,1:NX[t,run],m],
                    Xweights[t,1:NX[t,run],m],
                    y[1:t],
                    Nsteps,
                    R,
                    NX[t,run],
                    Priors,
                    acc
                )
                
                # Set new particle system for m
                
                # -----------------------------------------
                thetaParticles[:,m,t+1,run] = newpars
                Z[m,t,run] = newZ
                Xparticles[t,1:NX[t,run],m] = newXs
                Xweights[t,1:NX[t,run],m] = newXweights
                # -----------------------------------------
            end
            
            # Set weights back to equal
            ParamWeights[:,t] = 1.0
        else
            # if rejuvenation was not done at time t
            thetaParticles[:,:,t+1,run] = thetaParticles[:,:,t,run]
        end

        # Checking the acc. rate
        accept_rates[t,:,run] = [t,acc/(Nsteps*NT)]

        if do_we_move
            if accept_rates[t,2,run] < acc_threshold
                @printf("Acceptance rate below threshold %.2f\n",acc_threshold)
                c = c*2^(-5*(-accept_rates[t,2,run]))

                if NX[t,run]*1.5 > Np_x_max
                    println("Cannot add more x-particles")
                else
                    println("Adding more particles")
                    NX[t+1:end,run] = floor(NX[t,run]*1.5)
                    addParticles = 1
                end               
            else
                @printf("Acceptance rate %.4f\n",accept_rates[t,2,run])
            end
        end

        ######### exchange step
        if addParticles
            tempZ = zeros(NT,t)
            for m in 1:NT
                [Z_,X_,Xw_,_] = PF_call(y[1:t],thetaParticles[:,m,t+1,run],NX[t+1,run],1)
                Xparticles[1:t,1:NX[t+1,run],m] = X_
                Xweights[1:t,1:NX[t+1,run],m] = Xw_
                tempZ[m,1:t] = Z_'     
            end

            ratio = exp(tempZ[:,t]-Z[:,t,run])
            ParamWeights[:,t] = ratio.*ParamWeights[:,t]
            Z[:,1:t,run] = tempZ

            # update algorithm values
            ParamWeights[:,t] = ParamWeights[:,t]/sum(ParamWeights[:,t])
            AA = wsample(ParamWeights[:,t],NT)
            thetaParticles[:,:,t+1,run] = thetaParticles[:,AA,t+1,run]
            Z[:,t,run] = Z[AA,t,run]
            Xparticles[t,1:NX[t,run],:] = Xparticles[t,1:NX[t,run],AA]            
            Xweights[t,1:NX[t,run],:] = Xweights[t,1:NX[t,run],AA]
            ParamWeights[:,t] = 1
        end 
        
        # print the progress and ESS and norm(R) for monitoring
        @printf("RUN %d/%d, Iteration: %d/%d, ESS = %.2f, norm of cholcov = %.4f\n",run,nruns,t,T,ESS(t),norm(R));
    end
end