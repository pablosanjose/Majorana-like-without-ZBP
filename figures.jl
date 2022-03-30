using JLD2, CairoMakie, ColorSchemes

files = [
    ["Data/exp=4 - Vs=-70 - δV = (38, 38)/dos - (γ = 5, a0 = 5, Δ = 0.2) - (n = 1, τΓ = 8.0, Δcore = 0.0).jld2",
     "Data/exp=4 - Vs=-70 - δV = (55, 0)/conductance - (γ = 5, a0 = 5, Δ = 0.2) - (n = 1, τΓ = 8.0, Δcore = 0.0).jld2"],
    ["Data/exp=4 - Vs=-70 - δV = (50, 50)/dos - (γ = 5, a0 = 5, Δ = 0.2) - (n = 1, τΓ = 27.0, Δcore = 0.0).jld2",
     "Data/exp=4 - Vs=-70 - δV = (60, 0)/conductance - (γ = 5, a0 = 5, Δ = 0.2) - (n = 1, τΓ = 27.0, Δcore = 0.0).jld2"],
    ["Data/exp=4 - Vs=-11.9 - δV = (30, 30)/dos - (γ = 5, a0 = 6, Δ = 0.2) - (n = 1, τΓ = 12.0, Δcore = 0.0).jld2",
     "Data/exp=4 - Vs=-11.9 - δV = (12, 0)/conductance - (γ = 5, a0 = 6, Δ = 0.2) - (n = 1, τΓ = 12.0, Δcore = 0.0).jld2"]
]

function fig_theory(files, zscales)
    # fig = Figure(resolution = (1200, 1200), font =:sans)
    fig = Figure(resolution = (1100, 600), font = "CMU Serif Roman")
    local hmap
    for (col, colfiles) in enumerate(files), (row, file) in enumerate(colfiles)
        dict = load(file)
        data = sum(values(dict["mjdict"]))
        m = row == 1 ? zscales[row, col] * maximum(data) : col == 3 ? 1 : 1e-3
        data ./= m
        xs, ys = dict["xs"], real.(dict["ys"])
        zlims = row == 1 ? (0,1) : (0, zscales[row, col] * maximum(data))

        # data = hcat(data[:,size(data,2):-1:2], data)
        # ys = vcat(-ys[end:-1:2], ys)

        xlabel = row == 2 ? L"\Phi/\Phi_0" : ""
        ylabel = col == 1 ? (row == 1 ? "E (meV)" : "V (mV)") : ""
        label = row == 1 ? "LDOS (arb. units)" : col == 3 ? L"dI/dV\,\,(G_0)" : L"dI/dV\,\,(10^{-3} G_0)"

        ax = Axis(fig[row, 2*col-1]; xlabel, ylabel)

        row < 2 && hidexdecorations!(ax, ticks = false)
        col > 1 && hideydecorations!(ax, ticks = false)

        hmap = heatmap!(ax, xs, ys, data; colormap = :thermal, colorrange = zlims)
        cbar = Colorbar(fig, hmap, label = label, ticklabelsvisible = row == 2,
            labelpadding = 5,  flipaxisposition = true, width = 10,ticksize = 2,
            ticklabelpad = 5)
        fig[row, 2*col] = cbar
    end

    Label(fig[1, 1, TopLeft()], "a", padding = (-40, 0, -5, 0), font = "CMU Serif Bold", textsize = 20)
    Label(fig[1, 3, TopLeft()], "c", padding = (-20, 0, -5, 0), font = "CMU Serif Bold", textsize = 20)
    Label(fig[1, 5, TopLeft()], "e", padding = (-20, 0, -5, 0), font = "CMU Serif Bold", textsize = 20)
    Label(fig[2, 1, TopLeft()], "b", padding = (-40, 0, -20, 0), font = "CMU Serif Bold", textsize = 20)
    Label(fig[2, 3, TopLeft()], "d", padding = (-20, 0, -20, 0), font = "CMU Serif Bold", textsize = 20)
    Label(fig[2, 5, TopLeft()], "f", padding = (-20, 0, -20, 0), font = "CMU Serif Bold", textsize = 20)
    Label(fig[1, 1, Top()], "Small induced pairing", tellwidth=false, tellheight=false)
    Label(fig[1, 3, Top()], "Large induced pairing", tellwidth=false, tellheight=false)
    Label(fig[1, 5, Top()], "Majorana", tellwidth=false, tellheight=false)

    colgap!(fig.layout, 1, 5)
    colgap!(fig.layout, 3, 5)
    colgap!(fig.layout, 5, 5)
    rowgap!(fig.layout, 1, 10)

    return fig
end

function fig_theory_alt(files, zscales)
    fig = Figure(resolution = (1100, 850), font = "CMU Serif Roman")
    local hmap
    for (col, colfiles) in enumerate(files)
        for (row, file) in enumerate(colfiles)
            dict = load(file)
            data = sum(values(dict["mjdict"]))
            m = row == 1 ? zscales[row, col] * maximum(data) : col == 3 ? 2.0 : 2.0 * 1e-3 # G0 = 2e^2/h
            data ./= m
            xs, ys = dict["xs"], real.(dict["ys"])
            zlims = row == 1 ? (0,1) : (0, zscales[row, col] * maximum(data))

            # data = hcat(data[:,size(data,2):-1:2], data)
            # ys = vcat(-ys[end:-1:2], ys)

            # xlabel = row == 2 ? L"\Phi/\Phi_0" : ""
            xlabel = ""
            ylabel = col == 1 ? (row == 1 ? "E (meV)" : "V (mV)") : ""
            label = row == 1 ? "LDOS (arb. units)" : col == 3 ? L"dI/dV\,\,(G_0)" : L"dI/dV\,\,(10^{-3} G_0)"

            ax = Axis(fig[row, 2*col-1]; xlabel, ylabel)

            hidexdecorations!(ax, ticks = false)
            col > 1 && hideydecorations!(ax, ticks = false)

            # colormap = cgrad(ColorScheme(["#5a0097", "#ac1f71", "#de514c", "#f99c26", "#eefa1b"]), 4, categorical=false)
            # hmap = heatmap!(ax, xs, ys, data; colormap = :thermal, colorrange = zlims)
            hmap = heatmap!(ax, xs, ys, data; colorrange = zlims)
            cbar = Colorbar(fig, hmap, label = label, ticklabelsvisible = row == 2,
                labelpadding = 5,  flipaxisposition = true, width = 10,ticksize = 2,
                ticklabelpad = 5)
            fig[row, 2*col] = cbar

            if row == 2
                zmax = (17.0, 7.0, 1.1)[col]
                zmin = (-0.2, -0.1, -0.04)[col]
                ylabel = col == 3 ? L"dI/dV\,\,(G_0)" : L"dI/dV\,\,(10^{-3} G_0)"
                # Horizontal cuts
                xs = real.(dict["xs"])
                icut = 1 + (length(dict["ys"]) - 1) ÷ 2
                ys = clamp.(data[:, icut], 0, zmax)
                xlabel = L"\Phi/\Phi_0"
                ax = Axis(fig[3, 2*col-1]; xlabel, ylabel, height = 100, yaxisposition = :right)
                lines!(ax, xs, ys, color = :black)
                xlims!(ax, 0, 2.5)
                ylims!(ax, zmin, zmax)
                # Vertical cuts
                xs = real.(dict["ys"])
                icut1, icut2 = 1, 1 + round(Int, (length(dict["xs"]) - 1) * 1.0 / maximum(dict["xs"]))
                ys1, ys2 = data[icut1,:], data[icut2,:]
                xlabel = "V (mV)"
                ax = Axis(fig[4, 2*col-1]; xlabel, ylabel, height = 100, yaxisposition = :right)
                lines!(ax, xs, ys1; label = L"\Phi/\Phi_0 = 0")
                lines!(ax, xs, ys2; label = L"\Phi/\Phi_0 = 1")
                col == 1 && axislegend(ax; position = :ct)
                xlims!(ax, extrema(xs)...)
                zmax´ = col == 3 ? zmax : 2.2*zmax
                ylims!(ax, zmin, zmax´)
            end
        end
    end

    labels = ["a" "e" "i"; "b" "f" "j"; "c" "g" "k"; "d" "h" "l"]
    for col in 1:3, row in 1:4
        x = col == 1 ? -40 : -40
        y = row == 1 ? 0 : -20
        Label(fig[row, 2*col-1, TopLeft()], labels[row, col], padding = (x, 0, y, 0), font = "CMU Serif Bold", textsize = 20)
    end
    # Label(fig[1, 3, TopLeft()], "c", padding = (-10, 0, -5, 0), font = "CMU Serif Bold", textsize = 20)
    # Label(fig[1, 5, TopLeft()], "e", padding = (-10, 0, -5, 0), font = "CMU Serif Bold", textsize = 20)
    # Label(fig[2, 1, TopLeft()], "b", padding = (-40, 0, -20, 0), font = "CMU Serif Bold", textsize = 20)
    # Label(fig[2, 3, TopLeft()], "d", padding = (-10, 0, -20, 0), font = "CMU Serif Bold", textsize = 20)
    # Label(fig[2, 5, TopLeft()], "f", padding = (-10, 0, -20, 0), font = "CMU Serif Bold", textsize = 20)
    Label(fig[1, 1, Top()], "Small induced pairing", tellwidth=false, tellheight=false)
    Label(fig[1, 3, Top()], "Large induced pairing", tellwidth=false, tellheight=false)
    Label(fig[1, 5, Top()], "Majorana", tellwidth=false, tellheight=false)

    colgap!(fig.layout, 1, -50)
    colgap!(fig.layout, 2, 35)
    colgap!(fig.layout, 3, -50)
    colgap!(fig.layout, 4, 35)
    colgap!(fig.layout, 5, -50)
    rowgap!(fig.layout, 1, 5)
    rowgap!(fig.layout, 2, 5)
    rowgap!(fig.layout, 3, 5)

    return fig
end

function my_theme()
    my_colors = [colorant"#5a0097", colorant"#ac1f71", colorant"#de514c", colorant"#f99c26", colorant"#eefa1b"]
    my_discrete_colors = [colorant"#27f2ee", colorant"#0900ef"]
    colormap = cgrad(ColorScheme(my_colors))
    palette= (color = my_discrete_colors,)
    return Theme(; colormap, palette)
end


# f = fig_theory_alt(files, zscales)
f = with_theme(my_theme()) do
    fig_theory_alt(files, zscales)
end

save("figure5_v6.pdf", f)

f

