#region ## DEPENDENCIES AND PARAMETERS
# using MKL

using Parameters, Arpack, ProgressMeter, LinearAlgebra, SparseArrays, Quantica, RandomNumbers
using Quantica: GreensFunction, allsitepositions, nsites, orbitals

@with_kw struct Params @deftype Float64  # nm, meV
    m = 0.026
    ħ2ome = 76.1996
    a0 = 10
    t = ħ2ome/(2m*a0^2)
    Rcore = 60
    Rshell = 85
    Vmax = 0
    Vmin = -40
    Vmax_lead = Vmax
    Vmin_lead = Vmin
    Δ::ComplexF64 = 0.2
    Δcore = 0.0
    g = 12
    μBΦ0 = 119.6941183 # meV nm^2
    P = 919.7 # meV nm
    Δg = 417  # meV
    Δs = 390  # meV
    α = P^2/3 * (1/Δg^2 - 1/(Δg + Δs)^2)  # nm^2, around 1.19 default
    α0 = 0
    γ = 2
    Vexponent = 3
    echarge = 1
    L = 500 # the length, in nm, of the closed system
    δL = 0
    δR = 0
    δSL = δL/2
    δSR = δR/2
    δVL = 0
    δVR = 0
    maxlobe2 = 0.5
end

const σ0τz = @SMatrix[1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
const σ0τx = @SMatrix[0 0 1 0; 0 0 0 1; 1 0 0 0; 0 1 0 0]
const σ0τ0 = @SMatrix[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
const σzτ0 = @SMatrix[1 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
const σzτz = @SMatrix[1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
const σyτy = @SMatrix[0 0 0 -1; 0 0 1 0; 0 1 0 0; -1 0 0 0]
const σyτz = @SMatrix[0 -im 0 0; im 0 0 0; 0 0 0 im; 0 0 -im 0]
const σyτ0 = @SMatrix[0 -im 0 0; im 0 0 0; 0 0 0 -im; 0 0 im 0]
const σxτz = @SMatrix[0 1 0 0; 1 0 0 0; 0 0 0 -1; 0 0 -1 0]
const σxτ0 = @SMatrix[0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]
const σ0 = SA[1 0; 0 1]
const σx = SA[0 1; 1 0]
const σy = SA[0 -im; im 0]
const σz = SA[1 0; 0 -1];

#endregion

#region ## CYLINDER MODEL, DECOMPOSED IN mⱼ

build_cyl(; shell1D = false, slead = false, kw...) = build_cyl(Params(; kw...); shell1D, slead)

build_cyl(; slead = false, kw...) = build_cyl(Params(; kw...); slead)

function build_cyl(p::Params; slead = false)
    @unpack a0, t, Rcore, Rshell, Vmin, Vmax, Vmin_lead, Vmax_lead, Vexponent, Δ, γ, maxlobe2, g, μBΦ0, echarge, L, δL, δR, δSL, δSR, δVL, δVR, Δcore, α, α0 = p

    R´ = floor(Rcore/a0)*a0
    L´ = L + δSL + δSR

    # The radial p² is derived in https://doi.org/10.1016/j.cpc.2015.08.002 (Eq. (20) with l = 0 and constant m)
    # For constant mass it reads t * (2Fⱼ - Fⱼ₊₁ * ½(rⱼ + rⱼ₊₁)/√(rⱼrⱼ₊₁) - Fⱼ₋₁ * ½(rⱼ + rⱼ₋₁)/√(rⱼrⱼ₋₁)) =
    # = onsite(2t) - hopping((r, dr) -> t*r/√(r+dr/2)(r-dr/2)) where t = ħ²/(2ma₀²) and Fⱼ=Ψ(rⱼ)√rⱼ
    # Boundary condition: remove j = 0 and replace the j = 1 onsite by 3t/2
    p² = onsite(r -> σ0τz * t * ifelse(r[2] ≈ a0, 2.0 + 1.5, 2.0 + 2.0)) -
         hopping((r, dr) -> t * σ0τz * ifelse(iszero(dr[1]), r[2]/sqrt(r[2]^2 - 0.25*dr[2]^2), 1), range = a0)
    V(ρ, v0, v1) =  v0 + (v1-v0) * (ρ/R´)^Vexponent
    dϕ(ρ, v0, v1) = -(Vexponent / R´) * (v1-v0) * (ρ/R´)^(Vexponent-1) # ϕ=-V
    V(r) = smooth(r[1], (0, δL, L´-δR, L´), (V(r[2], Vmax_lead, Vmin_lead), V(r[2], Vmax, Vmin), V(r[2], Vmax_lead, Vmin_lead))) +
           barrier(r[1], (0, δL), δVL) + barrier(r[1], (L´-δR, L´), δVR)
    dϕ(r) = smooth(r[1], (0, δL, L´-δR, L´), (dϕ(r[2], Vmax_lead, Vmin_lead), dϕ(r[2], Vmax, Vmin), dϕ(r[2], Vmax_lead, Vmin_lead)))
    potential = onsite(r -> V(r) * σ0τz)
    rashba = hopping((r, dr) -> (α0 + α * dϕ(r)) * (im * dr[1] / a0^2) * σyτz)  # pz = -i∂z = [0 -i; i 0]/a0

    area_LP = pi * (0.5*Rcore + 0.5*Rshell)^2
    # eA/ħ = 1/2 * pi * n r/area_LP, where n = area_LP*B/Φ0 and Φ0 = h/2e
    # V_Zeeman = 1/2 g μB Φ0 n/area_LP
    zeeman! = @onsite!((o; n = 0) -> o + (0.5 * g * μBΦ0 * n / area_LP) * σzτ0)
    eAφ(r, n) = echarge * 0.5 * pi * n * r[2]/ area_LP
    mj(Z, n) = Z + ifelse(iseven(round(Int, n)), 0.5, 0.0)
    J(Z, n) = mj(Z, n)*σ0τ0 - 0.5*σzτ0 - 0.5*round(Int, n)*σ0τz
    gauge! = @onsite!((o, r; n = 0, Z = 0) -> o + (eAφ(r, n) * σ0τz + J(Z, n) / r[2])^2 * t * a0^2 * σ0τz)

    # Shell
    ΣS(Δ, ω) = -(ω * σ0τ0 + Δ * σ0τx) / sqrt(Δ^2 - complex(ω)^2)

    Δmin = 0.00001
    littleparks(n) = max(Δmin, 1 - γ * (n - round(n))^2 - 0.25*(1 - maxlobe2)*n^2)

    # Δsmooth(r) = smooth(r[1], (δSL, δL+δSL, L´-δR-δSR, L´-δSR), (0, 1, 0))
    Δsmooth(r) = smooth(r[1], (δSL, δSL, L´-δSR, L´-δSR), (0, 1, 0))
    ΣS! = @onsite!((o, r; ω = 0, τΓ = 1, n = 0) -> o + τΓ * Δ * Δsmooth(r) * ΣS(littleparks(n) * Δ, ω); region = r -> r[2] >= R´ - 0.1 * a0)
    ΣΔcore! = @onsite!((o, r; Δcore = 0, n = 0) -> o - littleparks(n) * Δcore * Δsmooth(r) * (r[2]/R´)^round(Int, n) * σ0τx; region = r -> r[2] < R´ - 0.1 * a0)
    # ΣS_lead! = @onsite!((o; ω = 0, τΓ = 1) -> o + τΓ * Δ * ΣS(0, ω); region = r -> r[2] >= R´ - 0.1 * a0)
    # ΣS_lead! = @onsite!((o; ω = 0, τΓ = 1, n = 0) -> o + τΓ * Δ * ΣS(littleparks(n) * Δ, ω); region = r -> r[2] >= R´ - 0.1 * a0)
    # ΣΔcore_lead! = @onsite!((o, r; Δcore = 0, n = 0) -> o - littleparks(n) * Δcore * (r[2]/R´)^round(Int, n) * σ0τx; region = r -> r[2] < R´ - 0.1 * a0)

    # Normal Lead
    latlead = LP.square(; a0) |> unitcell((1, 0), region = r -> 0 < r[2] <= R´)
    potential_lead = onsite(r -> V(r[2], Vmax_lead, Vmin_lead) * σ0τz)
    hlead = hamiltonian(latlead, p² + potential_lead + rashba, orbitals = Val(4))
    # phlead = slead ? hlead |> parametric(gauge!, zeeman!, ΣS_lead!, ΣΔcore_lead!) : hlead |> parametric(gauge!, zeeman!)
    phlead = hlead |> parametric(gauge!, zeeman!)

    # Normal Lead self-energy
    function ΣN(ω, side = 1; kw...)
        h´ = phlead(; ω, kw...)
        g0 = greens(h´, Schur1D(), boundaries = (0,))(ω, side=>side)
        return h´[(-side,)] * g0 * h´[(side,)]
    end

    # Central
    latL = LP.square(; a0) |> unitcell(region = r -> 0 <= r[1] <= L´ && 0 < r[2] <= R´)
    iC = siteindices(latL, region = r -> L´/2 <= r[1] <= L´/2+0.99a0 && 0 < r[2] <= R´) |> collect
    iLL = siteindices(latL, region = r -> δL <= r[1] <= δL+0.99a0 && 0 < r[2] <= R´) |> collect
    iL = siteindices(latL, region = r -> 0 <= r[1] < 0.99a0 && 0 < r[2] <= R´) |> collect
    iR = siteindices(latL, region = r -> L´-0.99a0 < r[1] <= L´ && 0 < r[2] <= R´) |> collect
    iS = siteindices(latL, region = r -> r[2] > R´ - 0.99*a0) |> collect
    ΣN0 = onsite(0I, indices = vcat(iL, iR, iS)) + hopping(0I, range = Inf, indices = (iR => iR, iL => iL))
    h0 = hamiltonian(latL, p² + potential + rashba + ΣN0, orbitals = Val(4))
    

    ΣR! = @block!((b; τR = 1, ω = 0, kw...) -> iszero(τR) ? b : b + τR * ΣN(ω, 1; kw...), iR)
    ΣL! = @block!((b; τL = 1, ω = 0, kw...) -> iszero(τL) ? b : b + τL * ΣN(ω, -1; kw...), iL)

    ph = h0 |> parametric(gauge!, zeeman!, ΣS!, ΣΔcore!, ΣR!, ΣL!)

    hΓ = hamiltonian(latL, ΣN0, orbitals = Val(4))
    ΓL! = @block!((b; τL = 1, ω = 0, kw...) -> iszero(τL) ? b : b + τL * img(ΣN(ω, -1; kw...)), iL)
    ΓL = (hΓ |> parametric(ΓL!))[:, iL]
    img(Σ) = (Σ'-Σ)/2im

    # Inverse Greens and lifetimes
    # ΓR! = @block!((b; τR = 1, ω = 0, kw...) -> iszero(τR) ? b : b + τR * img(ΣN(ω, 1; kw...)), iR)
    # ΓL! = @block!((b; τL = 1, ω = 0, kw...) -> iszero(τL) ? b : b + τL * img(ΣN(ω, -1; kw...)), iL)
    # ΓS! = @onsite!((o; ω = 0, τΓ = 1, n = 0) -> o + τΓ * Δ * img(ΣS(littleparks(n) * Δ, ω)); region = r -> r[2] >= R´ - 0.1 * a0  && δL/2 <= r[1] <= L´-δR/2)
    # img(Σ) = (Σ'-Σ)/2im
    # ΓL = (hΓ |> parametric(ΓL!))[:, iL]
    # ΓR = (hΓ |> parametric(ΓR!, ΓS!))[:, vcat(iR, iS)]
    # # ΓR = (hΓ |> parametric(ΓR!, ΓS!))[:, iR]
    # invg_onsite! = @onsite!((o; ω = 0) -> ω*σ0τ0 - o)
    # invg_hopping! = @hopping!(t -> -t)
    # ig = h0 |> parametric(gauge!, zeeman!, ΣS!, ΣΔcore!, ΣR!, ΣL!, invg_onsite!, invg_hopping!)

    return ph, iC, iLL, phlead, ΓL
end

function smooth(x, (x0, x1)::NTuple{2,Any}, (y0, y1)::NTuple{2,Any})
    x <= x0 ? y0 :
    x >= x1 ? y1 :
    y0 + (y1-y0) * (tanh(5*(x-0.5*(x0+x1))/abs(x1-x0))+1)/2
end

smooth(x, (x0, x1, x2)::NTuple{3,Any}, (y0, y1, y2)::NTuple{3,Any}) =
    smooth(x, (x0, x1), (y0, y1)) + smooth(x, (x1, x2), (0, y2-y1))
smooth(x, (x0, x1, x2, x3)::NTuple{4,Any}, (y0, y1, y2)::NTuple{3,Any}) =
    smooth(x, (x0, x1), (y0, y1)) + smooth(x, (x2, x3), (0, y2-y1))

barrier(x, (x0, x1), V) = x0 < x < x1 ? V*exp(-25*(x-0.5*(x0+x1))^2/(x1-x0)^2) : 0.0

#endregion

#region ## DOS

## Finite length, using greens
function dos(hf::Function, xrng, ωrng, L, ph)
    os = orbitalstructure(parent(ph))
    pts = Iterators.product(xrng, ωrng)
    d = @showprogress pmap(pts) do pt
        x, ω = pt
        h = hf(x, ω)
        g₀ = greens(h, Schur1D(), boundaries = (0,))
        g₀ω = g₀(ω)
        L´ = round(Int,L + 1)
        g₀_L´L´⁻¹ = Quantica.unflatten_blocks(inv(flatten(g₀ω[L´=>L´], os)), os)
        g11 = g₀ω[1=>1] - g₀ω[L´=>1] * g₀_L´L´⁻¹ * g₀ω[1=>L´]
        return -imag(tr(tr(g11)))/π
    end
    return reshape(d, size(pts)...)
end

## Semi-infinite, using greens
function dos(hf::Function, xrng, ωrng; kw...)
    pts = Iterators.product(xrng, ωrng)
    d = @showprogress pmap(pts) do pt
        x, ω = pt
        gf = greens(hf(x, ω), Schur1D(), boundaries = (0,))
        return -imag(tr(tr(gf(ω, 1=>1))))/π
    end
    return reshape(d, size(pts)...)
end

## Finite length, using \ (ldiv)
function dos_ldiv(hf::Function, xrng, ωrng, iL, ph)
    IL = identity_inds(iL, size(ph, 1))
    pts = Iterators.product(xrng, ωrng)
    x, ω = first(pts)
    m = similarmatrix(hf(x, ω), flatten)
    d = @showprogress pmap(pts) do pt
        x, ω = pt
        h = hf(x, ω)
        bloch!(m, h)
        IL´ = (ω*I - m) \ IL
        dos = 0.0
        for (c, i) in enumerate(iL), k in 1:4
            dos += -imag(IL´[4(i-1)+k, 4(c-1)+k])/π
        end
        return dos
    end
    return reshape(d, size(pts)...)
end

function identity_inds(iL, n)
    IL = fill(0.0, 4*n, 4*length(iL))
    for (c, i) in enumerate(iL), k in 1:4
        IL[4(i-1) + k, 4(c-1) + k] = 1
    end
    return IL
end

#endregion

#region ## CONDUCTANCE

function conductance(ph, Γ, nrng, ωrng; kw...)
    pts = Iterators.product(nrng, ωrng)
    hmat = similarmatrix(ph, flatten)
    Γmat = similarmatrix(Γ, Matrix{Quantica.blockeltype(Γ)})
    Γmat´ = similar(Γmat)
    iΓ = Quantica.axesflat(Γ, 2)
    τe, τz = SA[1,1,0,0],  SA[1,1,-1,-1]
    gs = @showprogress pmap(pts) do pt
        n, ω = pt
        conductance = 0.0

        bloch!(hmat, ph(; kw..., ω=ω, n=n))
        bloch!(Γmat, Γ(; kw..., ω=ω, n=n))
        copy!(Γmat´, Γmat)

        # Why is this faster than a straight lu(hmat) ???
        luig = lu(ω*I - hmat)
        GrΓ = view(ldiv!(luig, Γmat), iΓ, :)
        GaΓ = view(ldiv!(luig', Γmat´), iΓ, :)

        # G = 2i*Tr[(GrΓ - GaΓ)τe] + 4*Tr[GaΓ τh GrΓ τe] - 4*Tr[GaΓ τe GrΓ τe]
        #   = 2i*Tr[(GrΓ - GaΓ)τe] - 4*Tr[GaΓ τz GrΓ τe]
        conductance -= 4 * real(trace_product(GaΓ, τz, GrΓ, τe))
        GrΓ .-= GaΓ
        conductance += 2 * real(im * trace_product(GrΓ, τe))

        return conductance # The unit is e^2/h
    end
    return gs
end

function trace_product(GΓ::AbstractMatrix{T}, τ::SVector{N}) where {N,T}
    tp = zero(real(T))
    cols = size(GΓ, 2)
    for j in 1:cols
        tp += GΓ[j,j] * τ[mod1(j, N)]
    end
    return tp
end

function trace_product(GΓ1::AbstractMatrix{T}, τ1::SVector{N}, GΓ2, τ2::SVector{N}) where {N,T}
    rows, cols = size(GΓ1)
    tr = zero(real(T))
    for j in 1:cols, i in 1:rows
        tr += GΓ1[i, j] * τ1[mod1(j, N)] * GΓ2[j, i] * τ2[mod1(i, N)]
    end
    return tr
end

#endregion

#region ## CONDUCTANCE OLD


# ig = g(ω)⁻¹ = [ω - H0 - ΣL(ω) - ΣR(ω)]⁻¹, Γⱼ = (Σⱼ - Σⱼ')/2i
# Conductance: G = dI/dV = RA + TCAR + TEC
# RA   = 4Tr[ΓLᵉ g ΓLʰ g']
# TCAR = 4Tr[ΓLᵉ g ΓRʰ g']
# TEC  = 4Tr[ΓLᵉ g ΓLᵉ g']
"""
IL = -e/h ∫dω ∑_β Tr[(FL ΓL Gr Γβ Ga - ΓL Gr Fβ Γβ Ga)τz]
Fα = [f(ω-μα) 0 ; 0 f(ω + μα)]
∂μα Fα = [δ(ω-μα) 0 ; 0 -δ(ω + μα)] = σe δ(ω-μα) - σh δ(ω + μα)

∂μα Iα = e/h Tr[γRL´ˣᵉγLRᵉˣ + γLL´ᵉʰγLLʰᵉ + γLL´ʰᵉγLLᵉʰ](ω = eV) +
        +e/h Tr[γRL´ˣʰγLRʰˣ + γLL´ᵉʰγLLʰᵉ + γLL´ʰᵉγLLᵉʰ](ω = -eV)
where γαβˣʸ = [Gr_αβ Γβ]ˣʸ and γαβ'ˣʸ = [Ga_αβ Γβ]ˣʸ

Define Zαβˣ = Gr_αβ Γβ σx

"""
function conductance(ig, ΓL, ΓR, nrng, ωrng; kw...)
    pts = Iterators.product(nrng, ωrng)
    igmat = similarmatrix(ig, flatten)
    ΓRmat = similarmatrix(ΓR, Matrix{Quantica.blockeltype(ΓR)})
    ΓLmat = similarmatrix(ΓL, Matrix{Quantica.blockeltype(ΓL)})
    ΓLmat´ = similar(ΓLmat')
    iL = Quantica.axesflat(ΓL, 2)
    iR = Quantica.axesflat(ΓR, 2)
    e, h = Diagonal(SA[1,1,0,0]),  Diagonal(SA[0,0,1,1])
    gs = @showprogress pmap(pts) do pt
        n, ω = pt
        conductance = 0.0

        bloch!(igmat, ig(; kw..., ω=ω, n=n))
        bloch!(ΓLmat, ΓL(; kw..., ω=ω, n=n))
        bloch!(ΓRmat, ΓR(; kw..., ω=ω, n=n))

        luig = lu(igmat)
        copy!(ΓLmat´, ΓLmat')
        GΓR = ldiv!(luig, ΓRmat)
        GΓL = ldiv!(luig, ΓLmat)
        ΓLG = fake_rdiv!(ΓLmat´, luig)
        ZLR  = view(GΓR, iL, :)
        ZLR´ = view(ΓLG, :, iR)
        conductance += 2*contract(ZLR´, ZLR, (e, e), (e, h))
        ZLL  = view(GΓL, iL, :)
        ZLL´  = view(ΓLG, :, iL)
        conductance += 2*contract(ZLL´, ZLL, (e, h), (h, e))

        ## The -ω contributions seem to be always equal to the +ω contributions
        ## hence the 2* above
        
        # bloch!(igmat, ig(; kw..., ω=-ω, n=n))
        # bloch!(ΓLmat, ΓL(; kw..., ω=-ω, n=n))
        # bloch!(ΓRmat, ΓR(; kw..., ω=-ω, n=n))

        # luig = lu(igmat)
        # copy!(ΓLmat´, ΓLmat');
        # GΓR = ldiv!(luig, ΓRmat)
        # GΓL = ldiv!(luig, ΓLmat)
        # ΓLG = fake_rdiv!(ΓLmat´, luig)
        # ZLR  = view(GΓR, iL, :)
        # ZLR´ = view(ΓLG, :, iR)
        # conductance += contract(ZLR´, ZLR, (h, h), (h, e))
        # ZLL  = view(GΓL, iL, :)
        # ZLL´  = view(ΓLG, :, iL)
        # conductance += contract(ZLL´, ZLL, (e, h), (h, e))

        ## The -ω contributions seem to be always equal to the +ω contributions

        # bloch!(igmat, ig(; ω=ω, kw...))
        # bloch!(ΓLmat, ΓL(;  ω=ω, kw...))
        # bloch!(ΓRmat, ΓR(;  ω=ω, kw...))
        # luig = lu(igmat)
        # copy!(ΓLmat´, ΓLmat');
        # conductance += contract((iL, ΓLmat´), luig, (iR, ΓRmat), (e, e), (e, h))
        # copy!(ΓLmat´, ΓLmat');
        # conductance += contract((iL, ΓLmat´), luig, (iL, ΓLmat), (e, h), (h, e))

        # bloch!(igmat, ig(; ω=-ω, kw...))
        # bloch!(ΓLmat, ΓL(;  ω=-ω, kw...))
        # bloch!(ΓRmat, ΓR(;  ω=-ω, kw...))
        # luig = lu(igmat)
        # copy!(ΓLmat´, ΓLmat');
        # conductance += contract((iL, ΓLmat´), luig, (iR, ΓRmat), (h, h), (h, e))
        # copy!(ΓLmat´, ΓLmat');
        # conductance += contract((iL, ΓLmat´), luig, (iL, ΓLmat), (e, h), (h, e))

        return conductance/2 # The unit is e^2/h. The 1/2 for Nambu n = 1/2(ne-nh)
    end
    return gs
end

# Computes Tr[(p´ Γ_β G_{β,α})'(G_{α,β} Γ_β p)] = Tr[(Z´)'Z], 
# where α,β can be L/R, and p_i are e/h projectors
function contract(Z´, Z, (p1´, p1), (p2´, p2))
    N = nambudim(p1)
    T = eltype(Z)
    S = SMatrix{N,N,T,N*N}
    rows, cols = size(Z´, 1), size(Z, 2)
    cond = zero(real(T))
    for j in 1:N:cols, i in 1:N:rows
        z = S(view(Z, i:i+3, j:j+3))
        z´ = S(view(Z´, i:i+3, j:j+3))
        cond += real(dot(p1´ * z´, z * p1))
        cond += real(dot(p2´ * z´, z * p2))
    end
    return 4 * cond # The 4 compensates the 1/2im in Γ
end

nambudim(p::Diagonal{<:Any,<:SVector{N}}) where {N} = N

fake_rdiv!(r, f) = copy!(r, (f' \ r')')

shift(vs::AbstractVector, d) = [v .+ (i-1)*d for (i,v) in enumerate(vs)]

function conductanceR(ig, ΓL, nrng, ωrng; kw...)
    pts = Iterators.product(nrng, ωrng)
    igmat = similarmatrix(ig, flatten)
    # ΓLmat = similarmatrix(ΓL, Matrix{Quantica.blockeltype(ΓL)})
    ΓLmat = similarmatrix(ΓL, Matrix{Quantica.blockeltype(ΓL)})
    ΓLmat´ = similar(ΓLmat')
    iL = Quantica.axesflat(ΓL, 2)
    e, h = Diagonal(SA[1,1,0,0]),  Diagonal(SA[0,0,1,1])
    gs = @showprogress pmap(pts) do pt
        n, ω = pt
        conductance = 0.0

        bloch!(igmat, ig(; kw..., ω=ω, n=n))
        bloch!(ΓLmat, ΓL(; kw..., ω=ω, n=n))
        bloch!(ΓRmat, ΓR(; kw..., ω=ω, n=n))

        luig = lu(igmat)
        copy!(ΓLmat´, ΓLmat')
        GΓR = ldiv!(luig, ΓRmat)
        GΓL = ldiv!(luig, ΓLmat)
        ΓLG = fake_rdiv!(ΓLmat´, luig)
        ZLR  = view(GΓR, iL, :)
        ZLR´ = view(ΓLG, :, iR)
        conductance += 2*contract(ZLR´, ZLR, (e, e), (e, h))
        ZLL  = view(GΓL, iL, :)
        ZLL´  = view(ΓLG, :, iL)
        conductance += 2*contract(ZLL´, ZLL, (e, h), (h, e))

        return conductance/2 # The unit is e^2/h. The 1/2 for Nambu n = 1/2(ne-nh)
    end
    return gs
end

#endregion

#region ## PLOTTING AND EXPORTING

using VegaLite, CairoMakie, JLD2

function plotdata(dos::AbstractMatrix;
        xmax = 3, xlims = (0, xmax), ymax = 0.22, ylims = (0, ymax), zmin = 0, zmax = Inf, zlims = (zmin, zmax),
        xlabel = "n", ylabel = "V [mV]", labelbar = "DOS [a.u.]", aspect = 1.6, label = "", kw...)
    xs = range(xlims..., length = size(dos, 1))
    ys = range(ylims..., length = size(dos, 2))
    fig = Figure(resolution = (1200, round(Int, 1200/aspect)), font =:sans)
    ax = Axis(fig; aspect, xlabel, ylabel)
    dosc = clamp.(dos, zmin, zmax)
    hmap = heatmap!(xs, ys, dosc; colormap = :thermal, colorrange = zlims, kw...)
    cbar = Colorbar(fig, hmap, label = labelbar,
        labelpadding = 5,  flipaxisposition = false,
        ticklabelpad = 30)
    fig[1, 1] = ax
    fig[1, 2] = cbar
    Label(fig[0, :], string(label), textsize = 20)
    return fig
end

function plot_and_export(xs, ys, mjdict; filename, zfactor = 0.3, zfactor´ = zfactor, xlabel = "Γ/t", kw...)
    path = dirname(filename)
    path0 = pwd()

    try run(`mkdir -p $path`) catch end
    @save(filename*".jld2", xs, ys, mjdict, params, model, filename)
    
    data = values(mjdict)
    ms = keys(mjdict)
    p = plotdata(sum(data); xlims = extrema(xs), zmax = zfactor * maximum(sum(data)), 
                 ylims = extrema(real.(ys)), xlabel, label = "total", kw...);
    
    filelist = "0.tmp.pdf"
    save("$path/0.tmp.pdf", p); 

    counter = 0
    for (mj, d) in sort!(collect(zip(ms, data)))
        pn = plotdata(d; xlims = extrema(xs), zmax = zfactor´ * maximum(d),
                      ylims = extrema(real.(ys)), xlabel, label = "mj = $mj", kw...);
        counter += 1; save("$path/$counter.tmp.pdf", pn); filelist *= " $counter.tmp.pdf"
    end
    cd(path)
    try
        pdftk = "pdftk $filelist cat output out.pdf"
        run(`bash -c $pdftk`)
        run(`mv out.pdf $(basename(filename)).pdf`)
    catch
        throw(error())
    finally
        c = "rm *.tmp.pdf"
        run(`bash -c $c`)
        cd(path0)
    end

    return p
end

#endregion