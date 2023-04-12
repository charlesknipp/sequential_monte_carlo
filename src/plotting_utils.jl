export plot_histograms,construct_histograms
export quantile,estimated_trend,plot_state_trajectory

# plot the histogram of the estimated parameter (need to extend to IBIS)
function construct_histograms(
        smc::Sampler,
        var_names::Vector{VT},
        N::Int64;
        label::String = "",
        kwargs...
    ) where VT <: AbstractString
    # pass variable keyword arguments to the plot options
    @pgf opts = {}
    push!(opts,kwargs...)

    Θ  = vcat(sample(smc.θ,weights(smc.ω),N)'...)
    dθ = size(Θ,2)

    hists = []
    @pgf for i in 1:dθ
        θi = Axis(
            {"area style",yticklabels="",title=var_names[i]},
            PlotInc(
                {"ybar interval",mark="no",opts...},
                Table(fit(Histogram,Θ[:,i]))
            )
        )

        if label != ""
            push!(θi,LegendEntry(label))
        end

        push!(hists,θi)
    end

    return hists
end

function plot_histograms(
        hists::AbstractVector
    )
    num_plots = string(cld(length(hists),2))
    @pgf gp = GroupPlot(
        {
            group_style = {
                group_size = "2 by "*num_plots,
                vertical_sep = "1.5cm"
            }
        },
        hists...
    )

    return gp
end

# wrapper that plots the filtered states across time
function draw_plots(
        data::DataFrame,
        quantiles::Vector{XT},
        states::Vector{XT},
        label::Union{String,LaTeXString}
    ) where XT <: Number
    # properly index the above geometry for use with dates
    T = nrow(data)
    idx = vcat(1:T,T:-1:1,[1])

    # draw the plot using PGF to produce nice LaTeX figures
    @pgf Axis(
        {
            date_coordinates_in = "x",
            font = "\\footnotesize",
            width = "\\linewidth",
            height = "4.5cm",
            xticklabel = "\\year"
        },
        [
            Plot(
                {fill="gray",fill_opacity=0.35,draw="none"},
                Table("x"=>data.date[idx],"y"=>quantiles)
            ),
            Plot(
                {no_marks,color="black"},
                Table("x"=>data.date,"y"=>data[:,2])
            ),
            Plot(
                {no_marks,color="red"},
                Table("x"=>data.date,"y"=>states)
            )
        ],
        Legend(["",names(data)[2],label])
    )
end

function observation_dist(ibis::IBIS)
    # initialize values
    y = 0.0
    Σ = 0.0

    # get the predicted measurement
    for m in 1:ibis.M
        mod = ibis.model(ibis.θ[m])
        B,R = mod.B,mod.R

        ym = B*ibis.x[m]
        Σm = B*ibis.Σ[m]*B' + R

        y += ibis.ω[m]*ym[1]
        Σ += ibis.ω[m]*Σm[1,1]
    end

    return y,Σ
end

estimated_trend(ibis::IBIS) = observation_dist(ibis)[1]

function estimated_trend(smc::SMC)
    return sum(
        m -> smc.ω[m]*mean(observation(
            smc.model(smc.θ[m]),
            smc.w[m]'*smc.x[m]
        )),
        collect(1:smc.M)
    )
end

# rework this to spit out quantiles of the observation since it is consistently
# univariate. May require a predict step from the kalman filter
function quantile(
        ibis::IBIS,
        p::Union{PT,Vector{PT}}
    ) where PT <: Number
    sort!(p)

    y,Σ = observation_dist(ibis)

    return quantile(Normal(y,sqrt(Σ)),p)
end

# getting quantiles for samplers is a little trickier
function quantile(
        smc::SMC,
        p::Union{PT,Vector{PT}}
    ) where PT <: Number
    sort!(p)

    qs = zero(p)

    # integrate out the states
    for m in 1:smc.M
        mod = smc.model(smc.θ[m])
        xm  = smc.w[m]'*smc.x[m]    # double check this work...

        qs += smc.ω[m]*quantile(observation(mod,xm),p)
    end

    return qs
end


# plot the filtered distribution over time
function plot_state_trajectory(
        xs::Matrix{Float64},
        qs::Matrix{Float64}
    )

    n_probs,T  = size(qs)
    n_fills    = div(n_probs,2)

    cols = palette("YlGnBu",maximum([n_fills,3]))
    #cols = palette("Spectral",11)

    @pgf axis = Axis({
        font = "\\footnotesize",
        width = "\\linewidth",
        height = "0.5\\linewidth",
        xmin = 1,
        xmax = T,
    })

    @pgf for i in 1:n_fills
        # define the shape formed by the quantile
        quantile_geometry = vcat(
            qs[i,:],
            reverse(qs[n_probs-i+1,:]),
            qs[i,1]
        )
        fill = Plot(
            {fill=cols[n_fills+1-i],draw="none"},
            Table(
                x = vcat(1:T,T:-1:1,[1]),
                y = quantile_geometry
            )
        )
        lower_curve = Plot(
            {color=cols[n_fills+1-i],no_marks},
            Table(
                x = 1:T,
                y = qs[i,:]
            )
        )
        upper_curve = Plot(
            {color=cols[n_fills+1-i],no_marks},
            Table(
                x = 1:T,
                y = qs[n_probs-i+1,:]
            )
        )
        push!(axis,fill,lower_curve,upper_curve)
    end

    @pgf states = Plot(
        {no_marks,color="black"},
        Table(x=1:T,y=vec(xs))
    )

    push!(axis,states)

    return axis
end
