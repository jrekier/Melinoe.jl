# ── Free wobble / nutation eigenmodes (mantle · core · BMO) ───────────────────
#
# M(ω) = M0 + ω M1, affine in the eigenvalue; free modes satisfy det M(ω) = 0, i.e.
# ω = eigvals(M0, −M1). Three roots: Chandler (ω≈0), FCN and BMO mode (both ω≈−1).
# ω is dimensionless, in units of Ω, in the terrestrial frame.
#
# m̃_c and m̃_b are measured relative to the mantle, so core_abs = m̃+m̃_c and
# BML_abs = m̃+m̃_b. Rows are the whole body, the core and the BML, each divided by
# its own moment A_XΩ²; columns are m̃, m̃_c, m̃_b:
#
#   M11 = −e + ω + κ(1+ω)      M12 = (Ac/A+ξc)(1+ω)   M13 = (Ab/A+ξb)(1+ω)
#   M21 = (1+γc)ω              M22 = 1+ec + (1+βc)ω   M23 = δc ω
#   M31 = (1+γb)ω              M32 = δb ω             M33 = 1+eb + (1+βb)ω
#
# The M0 figure block is diagonal, diag(1+ec, 1+eb). The fluid off-diagonals δc, δb
# are pure deformation compliances tied by Â_c·δc = Â_b·δb, so D^{1/2}MD^{-1/2} with
# D = diag(1, Â_c, Â_b) is symmetric there, off-diagonal −√(δc·δb).
#
# There is no Σ_bc interface torque: at the static closure ∇P+ρ∇φ = −ρ∇Φ_diff
# pointwise, so that torque reduces to the figure coupling (C_X−A_X)Ω²·(wobble)
# already carried by (1+e_c), (1+e_b). A Δρg restoring requires the interface off
# its equipotential, i.e. the dynamic deflection W(σ) = U + φ₁/g.

# Compliances, moments and dynamical ellipticities for the nutation matrix.
function _nutation_setup(m::PlanetModel; core_layer, bmo_layer, Ω_SI)
    c   = bmo_compliances(m; core_layer, bmo_layer, Ω_SI)
    flt = flattening(m; Ω_SI)
    A,  e  = _region_moment_e(m, flt, 0.0, 1.0)
    Ac, ec = _region_moment_e(m, flt, 0.0, c.d)
    Ab, eb = _region_moment_e(m, flt, c.d, c.b)
    return (; c, e, ec, eb, AcA = Ac/A, AbA = Ab/A)
end

# K_bc, K_bm are interface drags (viscous, electromagnetic), the analogues of MHB's
# K_cmb: K_bc acts on m̃_b − m̃_c, K_bm on m̃_b. A drag −K·(that) on the BML enters the
# BML row (÷A_bΩ²) as +K·(that); K_bc's reaction enters the core row (÷A_cΩ²) as
# −(A_b/A_c)·K_bc·(m̃_b−m̃_c), while K_bm's acts on the mantle, which owns no row (row 1
# is whole-body, where internal torques cancel). K_bc → ∞ welds the BML to the core,
# K_bm → ∞ to the mantle. Complex K dissipates; K = 0 is free and inviscid.
function _nutation_matrices(s; K_bc = 0.0, K_bm = 0.0)
    c = s.c
    M0 = [ -s.e + c.κ    s.AcA + c.ξc   s.AbA + c.ξb ;
            0.0          1 + s.ec       0.0          ;
            0.0          0.0            1 + s.eb     ]
    M1 = [ 1 + c.κ       s.AcA + c.ξc   s.AbA + c.ξb ;
           1 + c.γc      1 + c.βc       c.δc         ;
           1 + c.γb      c.δb           1 + c.βb     ]
    if K_bc != 0 || K_bm != 0
        M0 = complex(M0); M1 = complex(M1)
        M0[3,3] += K_bc + K_bm
        M0[3,2] -= K_bc
        M0[2,2] += (s.AbA / s.AcA) * K_bc
        M0[2,3] -= (s.AbA / s.AcA) * K_bc
    end
    return M0, M1
end

"""
    nutation_modes(m; core_layer=1, bmo_layer=2, Ω_SI=Ω_EARTH, K_bc=0, K_bm=0)

Free wobble/nutation eigenmodes of the mantle–core–BMO system, from det M(ω)=0.
Returns `(; ω, T_cel, T_wob, e, ec, eb, AcA, AbA, c, M0, M1)`: `ω` the three
dimensionless eigenvalues (units of Ω, terrestrial frame), `T_cel = 2π/(Ω(1+ω))`
the celestial-frame nutation periods and `T_wob = 2π/(Ωω)` the terrestrial wobble
periods, both in days. `c` is the `bmo_compliances` result.
"""
function nutation_modes(m::PlanetModel; core_layer::Int = 1, bmo_layer::Int = 2, Ω_SI = nothing,
                        K_bc = 0.0, K_bm = 0.0)
    Ω_SI = rotation_rate(m, Ω_SI)
    s = _nutation_setup(m; core_layer, bmo_layer, Ω_SI)
    M0, M1 = _nutation_matrices(s; K_bc, K_bm)
    ω = eigvals(M0, -M1)                                   # det(M0 + ω M1) = 0
    day = 86400.0
    T_cel = [2π/(Ω_SI*(1 + ωi))/day for ωi in ω]
    T_wob = [2π/(Ω_SI*ωi)/day       for ωi in ω]
    return (; ω, T_cel, T_wob, s.e, s.ec, s.eb, s.AcA, s.AbA, s.c, M0, M1)
end

"""
    ekman_K_bc(m; core_layer=1, bmo_layer=2, Ω_SI=nothing, ν_SI)

Viscous (Ekman-layer) estimate of the Σ_bc coupling `K_bc` for a BML of thickness
`h`: the boundary layers on the shell's two faces drag it toward the core,
`K_bc ≈ (3/2)·δ/h` with the Ekman depth `δ = √(ν/Ω)`, so a thinner shell couples more
strongly. `ν_SI` is the kinematic viscosity [m²/s]: silicate magma ~1e-3–1e0, a
crystal-rich mush higher.
"""
function ekman_K_bc(m::PlanetModel; core_layer::Int = 1, bmo_layer::Int = 2,
                    Ω_SI = nothing, ν_SI::Real)
    Ω_SI = rotation_rate(m, Ω_SI)
    d = rightendpoint(m.layers[core_layer].domain)
    b = rightendpoint(m.layers[bmo_layer].domain)
    h = (b - d) * m.R_SI                       # shell thickness [m]
    δ = sqrt(ν_SI / Ω_SI)                      # Ekman depth [m]
    return 1.5 * δ / h
end

"""
    ekman_K_bm(m; core_layer=1, bmo_layer=2, Ω_SI=nothing, ν_SI)

Viscous estimate of the Σ_bm coupling `K_bm`, the drag on the BML at its **outer**
face. Same scaling as [`ekman_K_bc`] over one boundary layer instead of two,
`K_bm ≈ (3/4)·δ/h`, so `ekman_K_bc = ekman_K_bm(inner) + ekman_K_bm(outer)`.

`ν_SI` is the viscosity of the magma; the solid above enters as a no-slip wall, valid
while its Maxwell time `η/μ` exceeds the period of the driving shear (the BML–mantle
differential rotation, near-diurnal in the terrestrial frame).
"""
function ekman_K_bm(m::PlanetModel; core_layer::Int = 1, bmo_layer::Int = 2,
                    Ω_SI = nothing, ν_SI::Real)
    Ω_SI = rotation_rate(m, Ω_SI)
    d = rightendpoint(m.layers[core_layer].domain)
    b = rightendpoint(m.layers[bmo_layer].domain)
    h = (b - d) * m.R_SI
    δ = sqrt(ν_SI / Ω_SI)
    return 0.75 * δ / h
end

"""
    nutation_transfer(m, ω; core_layer=1, bmo_layer=2, Ω_SI=Ω_EARTH, K_bc=0, K_bm=0)

Forced-nutation transfer function `T(ω) = m̃/m̃_rigid` at real tidal frequency `ω`
(units of Ω): the whole-body wobble over the single-rigid-body response
`m̃_rigid = eφ/(ω−e)`. Solves `M(ω)·x = b(ω)` with `b = [e − κ(1+ω), −γc ω, −γb ω]`
and returns `x₁·(ω−e)/e`. `ω` may be a scalar or a vector; the poles are the free
modes.
"""
function nutation_transfer(m::PlanetModel, ω; core_layer::Int = 1, bmo_layer::Int = 2,
                           Ω_SI = nothing, K_bc = 0.0, K_bm = 0.0)
    Ω_SI = rotation_rate(m, Ω_SI)
    s = _nutation_setup(m; core_layer, bmo_layer, Ω_SI); c = s.c; e = s.e
    M0, M1 = _nutation_matrices(s; K_bc, K_bm)
    _T(w) = begin
        b = [e - c.κ*(1 + w), -c.γc*w, -c.γb*w]
        ((M0 + w*M1) \ b)[1] * (w - e)/e
    end
    return ω isa Number ? _T(ω) : _T.(ω)
end

"""
    nutation_residues(m; core_layer=1, bmo_layer=2, Ω_SI=nothing, K_bc=0, K_bm=0)

Resonance strengths of the forced-nutation transfer function: the numerators `N_j`
in the pole expansion

    T(ω) = q₀ + q₁ω + Σ_j N_j/(ω − ω_j) ,      N_j = lim_{ω→ω_j} (ω − ω_j) T(ω)

Returns `(; ω, N, F, q)`, the eigenvalues ordered as `nutation_modes` returns them
(sort by `real` for `[FCN, BML, Chandler]`), `q = (q₀, q₁)` the non-resonant
background (linear: `T` is a quartic over a cubic), and `N`, `ω` in units of Ω.

`F = N ./ (1 .+ ω)` are the same strengths in the amplification-factor
normalization, which writes the transfer function against the celestial frequency
`ν = Ω(1+ω)`:

    T(ν) = 1 + Σ_j F_j · ν/(ν − ν_j) ,        ν_j = 1 + ω_j

so `F_j = N_j/ν_j`, residues being invariant under the shift. For a BML-free
two-layer body this reduces to `F = A_f/(A−A_f)(1 − γ/e)` and
`ν_FCN = −A/(A−A_f)(e_f − β)`.

`N_j` is evaluated in closed form: writing `T` by Cramer as
`(ω−e)/e · det M⁽¹⁾(ω)/det M(ω)`, with column 1 replaced by the forcing `b`, and
using `det M(ω) = det(M1)·∏_k (ω − ω_k)`,

    N_j = (ω_j − e)/e · det M⁽¹⁾(ω_j) / [ det(M1) · ∏_{k≠j} (ω_j − ω_k) ] .

A mode's effect on a given nutation scales as `N_j/(ω − ω_j)`, so a large residue at
a remote pole can still be negligible.
"""
function nutation_residues(m::PlanetModel; core_layer::Int = 1, bmo_layer::Int = 2,
                           Ω_SI = nothing, K_bc = 0.0, K_bm = 0.0)
    Ω_SI = rotation_rate(m, Ω_SI)
    s = _nutation_setup(m; core_layer, bmo_layer, Ω_SI); c = s.c; e = s.e
    M0, M1 = _nutation_matrices(s; K_bc, K_bm)
    ω = eigvals(M0, -M1)
    bvec(w) = [e - c.κ*(1 + w), -c.γc*w, -c.γb*w]
    N = map(eachindex(ω)) do j
        Mj = complex(M0 + ω[j]*M1); Mj[:, 1] = bvec(ω[j])       # M⁽¹⁾(ω_j)
        (ω[j] - e)/e * det(Mj) / (det(M1) * prod(ω[j] - ω[k] for k in eachindex(ω) if k != j))
    end
    q₁ = (M1 \ [-c.κ, -c.γc, -c.γb])[1] / e                     # T ~ q₁ω as ω→∞
    ω★ = 0.5 + maximum(abs, ω)                                  # a point off every pole
    T★ = ((M0 + ω★*M1) \ bvec(ω★))[1] * (ω★ - e)/e
    q₀ = T★ - q₁*ω★ - sum(N[k]/(ω★ - ω[k]) for k in eachindex(ω))
    return (; ω, N, F = N ./ (1 .+ ω), q = (q₀, q₁))
end
