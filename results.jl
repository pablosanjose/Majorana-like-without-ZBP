#region ## LOAD CODE (MULTICORE)
using Distributed
const NWORKERS = 16
nprocs() <= NWORKERS && addprocs(NWORKERS + 1 - nprocs())
@everywhere begin
    using Pkg
    Pkg.activate(".")
    using LinearAlgebra
    BLAS.set_num_threads(1)
end

@everywhere begin
    using Revise
    includet("code.jl")
end

#endregion

#region ## BANDS & LDOS

## Bandstructure
hh = phlead(; Z = 9, ω = 0, n = 1, τΓ = 0.0)
# hh = phshell1D(Δ = 0.2)
b = bandstructure(hh, cuboid((0, 0.5 * model.a0), subticks = 81), method = ArpackPackage(nev = 36, sigma = 0.1im))
vlplot(b, size = 400, ylims = (-50,50))

## Eigenstates
(s -> s.energy).(b[(0,), around = (0, 12)])
Fj = b[(0,), around = (0, 12)][8].basis[:,1]
psi = Fj ./ sqrt.(1:length(Fj))
vlplot(hh, psi, sitesize = DensityShader(), maxthickness = 0.01)

## Extended LDOS

hh = phlead(; params..., Z = 0, ω = 0, n = 1)
g = greens(hh, Schur1D(), boundaries = (0,))
g0 = g(0.000+0.00001im)
ldos = [(n,) => -imag.(diag.(diag(g0[n=>n]))) for n in 1:100];
vlplot(hh, ldos, sitecolor = DensityShader(), sitesize = DensityShader(), maxdiameter = 14, mindiameter = 0, plotlinks = false)

## Lattice
vlplot(ph(; params..., Z = 4, ω = 0.1+0.00001im, n = 1), maxdiameter = 5, plotlinks = false)

vlplot(ph(; params..., Z = 4, ω = 0.1+0.00001im, n = 1), maxdiameter = 17, xlims = (750, 800))

vlplot(phlead(; params..., Z = 4, ω = 0.1+0.00001im, n = 1), maxdiameter = 17, xlims = (0, 50))

#endregion

#region ## LOAD CONFIGURATION AND DENSITY RESULTS

@load "Output/device_C/L=800 - exp=6 - R=80 - Vs=-12 - δV = (40, 0)/conductance - (γ = 4.5, a0 = 8, m = 0.1, Δ = 0.2) - (n = 1, τΓ = 1.0, Δcore = 0.2).jld2"
# plot_and_export(xs, ys, mjdict; filename, zfactor = 0.5, xlabel = "Φ/Φ₀", labelbar = "dI/dV [e²/h]")
# plot_and_export(xs, ys, mjdict; filename, zfactor = 0.7, xlabel = "Φ/Φ₀", labelbar = "LDOS [a.u.]")
#endregion

#region ## CONFIGURATION A 
#region ## close/close (LDOS)

model = (;
    γ = 5, Rcore = 60, Rshell = 85, a0 = 5, L = 700, 
    Δ = 0.2, α=0, Vexponent = 4,
    Vmin = -70, Vmax = 0,
    Vmin_lead = -70, Vmax_lead = 0,
    δL = 95, δVL = 38, δR = 95, δVR = 38,
    δSL = 45, δSR = 45)
params = (; n = 1, τΓ = 8.0, Δcore = 0.0)

mjs = -8:8
ωrng = range(-0.23, 0.23, 401) .+ 0.001im
nrng = range(0, 2.499, 400)

ph, iC, iL, phlead, ΓL = build_cyl(; model...);

nω_hfunc(Z) = (n, ω)  -> ph(; params..., Z = Z, ω = ω, n)

## LDOS(n, ω, L)
folder = "Output/device_A/exp=$(model.Vexponent) - Vs=$(model.Vmin) - δV = $((model.δVL, model.δVR))"
filename = "$folder/dos - $(model[(:γ, :a0, :Δ)]) - $params"
xs, ys = nrng, ωrng;
mjdict = Dict([(@show Z; Z => dos_ldiv(nω_hfunc(Z), xs, ys, iL, ph)) for Z in mjs]);
p = plot_and_export(xs, ys, mjdict; filename, zfactor = 0.5, xlabel = "Φ/Φ₀", labelbar = "LDOS [a.u]")
#endregion

#region ## Alt: open/open (dI/dV) - more shunting

model = (;
    γ = 5, Rcore = 60, Rshell = 85, a0 = 5, L = 700, 
    Δ = 0.2, α=0, Vexponent = 4,
    Vmin = -70, Vmax = 0,
    Vmin_lead = -70, Vmax_lead = 0,
    δL = 95, δVL = 55, δR = 95, δVR = 0,
    δSL = 43, δSR = 43)
params = (; n = 1, τΓ = 8.0, Δcore = 0.0)


mjs = -8:8
ωrng = range(-0.23, 0.23, 201) .+ 0.00001im
nrng = range(0, 2.499, 200)

ph, iC, iL, phlead, ΓL = build_cyl(; model...);

nω_hfunc(Z) = (n, ω)  -> ph(; params..., Z = Z, ω = ω, n)

## CONDUCTANCE(n, ω, L)
folder = "Output/device_A_alt/exp=$(model.Vexponent) - Vs=$(model.Vmin) - δV = $((model.δVL, model.δVR))"
filename = "$folder/conductance - $(model[(:γ, :a0, :Δ)]) - $params"
xs, ys = nrng, ωrng;
mjdict = Dict([(@show Z; Z => conductance(ph, ΓL, xs, ys; params..., Z)) for Z in mjs]);
p = plot_and_export(xs, ys, mjdict; filename, zfactor = 0.07, xlabel = "Φ/Φ₀", labelbar = "dI/dV [e²/h]")
#endregion

#endregion

#region ## CONFIGURATION C (like A but more pairing)

#region ## semi/semi (LDOS)

model = (;
    γ = 5, Rcore = 60, Rshell = 85, a0 = 5, L = 700, 
    Δ = 0.2, α=0, Vexponent = 4,
    Vmin = -70, Vmax = 0,
    Vmin_lead = -70, Vmax_lead = 0,
    δL = 95, δVL = 50, δR = 95, δVR = 50,
    δSL = 45, δSR = 45
    )
params = (; n = 1, τΓ = 27.0, Δcore = 0.0)

mjs = -8:8
ωrng = range(-0.23, 0.23, 401) .+ 0.001im
nrng = range(0, 2.499, 400)

ph, iC, iL, phlead, ΓL = build_cyl(; model...);

nω_hfunc(Z) = (n, ω)  -> ph(; params..., Z = Z, ω = ω, n)

## LDOS(n, ω, L)
folder = "Output/device_C/exp=$(model.Vexponent) - Vs=$(model.Vmin) - δV = $((model.δVL, model.δVR))"
filename = "$folder/dos - $(model[(:γ, :a0, :Δ)]) - $params"
xs, ys = nrng, ωrng;
mjdict = Dict([(@show Z; Z => dos_ldiv(nω_hfunc(Z), xs, ys, iL, ph)) for Z in mjs]);
p = plot_and_export(xs, ys, mjdict; filename, zfactor = 0.7, xlabel = "Φ/Φ₀", labelbar = "LDOS [a.u]");
p
#endregion

#region ## close/open (dI/dV)

model = (;
    γ = 5, Rcore = 60, Rshell = 85, a0 = 5, L = 700, 
    Δ = 0.2, α=0, Vexponent = 4,
    Vmin = -70, Vmax = 0,
    Vmin_lead = -70, Vmax_lead = 0,
    δL = 95, δVL = 60, δR = 95, δVR = 0,
    δSL = 45, δSR = 45
    )
params = (; n = 1, τΓ = 27.0, Δcore = 0.0)


mjs = -8:8
ωrng = range(-0.23, 0.23, 201) .+ 0.00001im
nrng = range(0, 2.499, 200)

ph, iC, iL, phlead, ΓL = build_cyl(; model...);

nω_hfunc(Z) = (n, ω)  -> ph(; params..., Z = Z, ω = ω, n)

## CONDUCTANCE(n, ω, L)
folder = "Output/device_C/exp=$(model.Vexponent) - Vs=$(model.Vmin) - δV = $((model.δVL, model.δVR))"
filename = "$folder/conductance - $(model[(:γ, :a0, :Δ)]) - $params"
xs, ys = nrng, ωrng;
mjdict = Dict([(@show Z; Z => conductance(ph, ΓL, xs, ys; params..., Z)) for Z in mjs]);
p = plot_and_export(xs, ys, mjdict; filename, zfactor = 0.05, xlabel = "Φ/Φ₀", labelbar = "dI/dV [e²/h]")
#endregion

#endregion

#region ## CONFIGURATION Majorana

#region ## semi/semi (LDOS)

model = (;
    γ = 5, Rcore = 60, Rshell = 85, a0 = 6, L = 700, 
    Δ = 0.2, g = 12, α0 = 22, α=0, Vexponent = 4,
    Vmin = -11.9, Vmax = 0,
    Vmin_lead = -100, Vmax_lead = 0,
    δL = 96, δVL = 30, δR = 96, δVR = 30)
params = (; n = 1, τΓ = 12.0, Δcore = 0.0)


mjs = -2:2
ωrng = range(-0.23, 0.23, 401) .+ 0.001im
nrng = range(0, 2.499, 400)

ph, iC, iL, phlead, ΓL = build_cyl(; slead = false, model...);
nω_hfunc(Z) = (n, ω)  -> ph(; params..., Z = Z, ω = ω, n)

## LDOS(n, ω, L)
folder = "Output/device_M/exp=$(model.Vexponent) - Vs=$(model.Vmin) - δV = $((model.δVL, model.δVR))"
filename = "$folder/dos - $(model[(:γ, :a0, :Δ)]) - $params"
xs, ys = nrng, ωrng;
mjdict = Dict([(@show Z; Z => dos_ldiv(nω_hfunc(Z), xs, ys, iL, ph)) for Z in mjs]);
plot_and_export(xs, ys, mjdict; filename, zfactor = .3, xlabel = "Φ/Φ₀", labelbar = "LDOS [a.u]")

#endregion

#region ## close/open (dI/dV)

model = (;
    γ = 5, Rcore = 60, Rshell = 85, a0 = 6, L = 700, 
    Δ = 0.2, g = 12, α0 = 22, α=0, Vexponent = 4,
    Vmin = -11.9, Vmax = 0,
    Vmin_lead = -100, Vmax_lead = 0,
    δL = 96, δVL = 12, δR = 96, δVR = 0)
params = (; n = 1, τΓ = 12.0, Δcore = 0.0)

mjs = -2:2
ωrng = range(-0.23, 0.23, 201) .+ 0.000005im
nrng = range(0, 2.499, 200)

ph, iC, iL, phlead, ΓL = build_cyl(; slead = false, model...);
nω_hfunc(Z) = (n, ω)  -> ph(; params..., Z = Z, ω = ω, n)

## CONDUCTANCE(n, ω, L)
folder = "Output/device_M/exp=$(model.Vexponent) - Vs=$(model.Vmin) - δV = $((model.δVL, model.δVR))"
filename = "$folder/conductance - $(model[(:γ, :a0, :Δ)]) - $params"
xs, ys = nrng, ωrng;
mjdict = Dict([(@show Z; Z => conductance(ph, ΓL, xs, ys; params..., Z)) for Z in mjs]);
plot_and_export(xs, ys, mjdict; filename, zfactor = 0.17, xlabel = "Φ/Φ₀", labelbar = "dI/dV [e²/h]")

#endregion

#endregion
