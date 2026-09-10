# ── Ganymede ──────────────────────────────────────────────────────────────────
# Differentiated icy moon: liquid Fe-FeS core, silicate mantle, high-pressure ice,
# ocean, ice shell.

"""
    Ganymede(; h_ice_km=100.0, n_core=50, n_solid=40) -> PlanetModel

Liquid Fe-FeS core, silicate mantle, high-pressure ice, ocean `layers[4]`, ice shell.
`h_ice_km` sets the shell thickness above the ocean floor, fixed at 2284 km.

| quantity | value | source |
|---|---|---|
| mass | 1.482e23 kg | Anderson et al. 1996, *Nature* **384**, 541 |
| polar MoI `C_nd` | 0.3105 | Anderson et al. 1996 |
| mean radius | 2631.2 km | Baland & Van Hoolst 2010, *Icarus* **209**, 651 |
| mean density | 1942 kg/m³ | Baland & Van Hoolst 2010 |
| rigidity: shell / HP ice / mantle | 3.3 / 6.6 / 50 GPa | Hussmann et al. 2016, *CMDA* **126**, 131 |
| orbital period | 7.155 d | Sohl et al. 2002, *Icarus* **157**, 104 |

Mass and `C_nd` are matched exactly; `k₂ = 0.463` and `h₂ = 1.369` then land inside
the published spreads for ocean-bearing Ganymede models (Macrì & Casotto 2025,
*A&A* **699**, A261, Fig. A.1), which also collect the values above.

**Two parameterisations.** By default the radii are frozen and the core and mantle
densities are scaled to hit `M` and `C_nd`, so the shell thickness is paid for in core
density: 7869 kg/m³ as `h_ice_km → 0`, 5991 at 84 km, 4996 at 134 km, and *lighter than
the mantle* past **219.1 km**, which throws (outside 4500–7500 you get a warning). That
is convenient for a single model but a poor family to sweep: `R_core` cannot move, so
all the adjustment lands on density.

Pass `ρ_core` and/or `ρ_mantle` for the **standard** parameterisation instead —
densities from material physics, radii solved from `M` and `C_nd`, as in Sohl,
Hussmann, Vance and the Macrì & Casotto ensemble. With `ρ_core = 5500`,
`ρ_mantle = 3300` the radii stay sensible across the whole hydrosphere range
(`R_core` 828 → 559 km, `R_mantle` 1791 → 1881 km over `h_ice_km` 20 → 330) and
nothing is strained. Prefer it whenever `h_ice_km` is a swept variable. The two agree
closely on rotational quantities — free periods match to ≲1.5% — because `M` and
`C_nd` already pin the moments the modes depend on.

`C_nd` is less certain than its formal error: it comes from Darwin–Radau, which
non-hydrostatic stress can inflate by over 10% (Gao & Stevenson 2013, *Icarus*
**226**, 1185), and the Juno reanalysis points to a larger true value (Gomez Casajus
et al. 2022, *GRL* **49**, e2022GL099475).

`pressure_love(m; forcing_layer = 4)` gives the ocean's pressure Love numbers.
"""
function Ganymede(; h_ice_km::Float64 = 100.0, n_core::Int = 50, n_solid::Int = 40,
                  ρ_core = nothing, ρ_mantle = nothing, Cnd = 0.3105,
                  μ_shell = 3.3e9)
    R   = 2631.2e3                      # mean radius, Baland & Van Hoolst 2010
    Rc  = 720.0e3; Rm = 1820.0e3; Rhp = 2284.0e3
    Ro  = R - h_ice_km*1e3

    # C_nd is far less certain than its formal error: Darwin–Radau can overestimate it
    # by >10% under non-hydrostatic stress (Gao & Stevenson 2013) and the Juno reanalysis
    # points to a larger value (Gomez Casajus et al. 2022). Macrì & Casotto (2025) decline
    # to impose it at all, sampling 0.25–0.35. Pass `Cnd` to explore that.
    M_obs, Cnd_obs = 1.482e23, Cnd           # Anderson et al. 1996, Nature 384, 541
    ρ_hyd = (1346.0, 1100.0, 937.0)     # HP ice, ocean, shell
    ρ_ref = (5777.9, 3291.5)            # core, mantle — starting profile

    if ρ_core !== nothing || ρ_mantle !== nothing
        # ── standard parameterisation: densities from material physics, radii free.
        # Two unknowns (Rc, Rm) against two constraints (M, C/MR²); Newton on both.
        ρc = something(ρ_core, ρ_ref[1]); ρm = something(ρ_mantle, ρ_ref[2])
        ρ_SI = (ρc, ρm, ρ_hyd...)
        function _res(a, b)
            e = (0.0, a, b, Rhp, Ro, R)
            (sum(4π/3 *(e[i+1]^3 - e[i]^3)*ρ_SI[i] for i in 1:5) - M_obs,
             sum(8π/15*(e[i+1]^5 - e[i]^5)*ρ_SI[i] for i in 1:5) - Cnd_obs*M_obs*R^2)
        end
        for _ in 1:200
            f1, f2 = _res(Rc, Rm); d = 1e3
            a1, a2 = _res(Rc+d, Rm); b1, b2 = _res(Rc, Rm+d)
            δ = [(a1-f1)/d (b1-f1)/d; (a2-f2)/d (b2-f2)/d] \ [-f1, -f2]
            Rc += δ[1]; Rm += δ[2]
            max(abs(δ[1]), abs(δ[2])) < 1e-6 && break
        end
        (0 < Rc < Rm < Rhp) || throw(ArgumentError(
            "Ganymede: no core/mantle radii satisfy M and C/MR² for h_ice_km = $h_ice_km " *
            "with ρ_core = $ρc, ρ_mantle = $ρm (got Rc = $(round(Rc/1e3,digits=1)) km, " *
            "Rm = $(round(Rm/1e3,digits=1)) km against Rhp = $(Rhp/1e3) km)."))
    else
        # ── frozen-radii parameterisation (default): radii fixed, densities scaled.
        edges = (0.0, Rc, Rm, Rhp, Ro, R)
        v3(i) = 4π/3 * (edges[i+1]^3 - edges[i]^3)      # mass weight
        v5(i) = 8π/15 * (edges[i+1]^5 - edges[i]^5)     # polar-MoI weight
        rest3 = sum(v3(i)*ρ_hyd[i-2] for i in 3:5)
        rest5 = sum(v5(i)*ρ_hyd[i-2] for i in 3:5)
        α, β = [v3(1)*ρ_ref[1]  v3(2)*ρ_ref[2]
                v5(1)*ρ_ref[1]  v5(2)*ρ_ref[2]] \ [M_obs - rest3, Cnd_obs*M_obs*R^2 - rest5]
        ρ_SI = (α*ρ_ref[1], β*ρ_ref[2], ρ_hyd...)            # ≈ 5662, 3313, …

        # Here `h_ice_km` is not a free knob: M and C/MR² are pinned and the radii
        # cannot move, so the shell thickness is paid for entirely in core density.
        # Past 219.1 km the "core" comes out lighter than the mantle above it —
        # gravitationally unstable, and every downstream quantity (flattening, Love
        # numbers, compliances) is then meaningless. Pass `ρ_core`/`ρ_mantle` to sweep.
        if !(4500 ≤ ρ_SI[1] ≤ 7500)
            @warn "Ganymede: h_ice_km = $h_ice_km forces ρ_core = $(round(ρ_SI[1], digits=0)) \
kg/m³, outside the 4500–7500 range an Fe–FeS core could plausibly span (Fe–FeS is 5000–6000, \
pure iron ≈ 7900). The radii are frozen, so shell thickness is paid for in core density — \
pass ρ_core/ρ_mantle for the radii-free parameterisation instead." maxlog = 1
        end
    end

    # gravitational stability: density must not increase outward, in either branch
    if !issorted(collect(ρ_SI); rev = true)
        throw(ArgumentError(
            "Ganymede: h_ice_km = $h_ice_km gives a density inversion (ρ_core = " *
            "$(round(ρ_SI[1], digits=0)) ≤ ρ_mantle = $(round(ρ_SI[2], digits=0)) kg/m³). " *
            "With the default frozen radii this caps the shell at 219.1 km; pass " *
            "ρ_core/ρ_mantle to solve for the radii instead, which has no such cap."))
    end

    μ_SI = (0.0, 50.0e9, 6.6e9, 0.0, μ_shell)                # Hussmann et al. 2016, CMDA 126, 131
    κ_SI = (Inf, 130.0e9, 14.0e9, Inf, 9.0e9)                # fluids incompressible

    xb = (Rc, Rm, Rhp, Ro, R) ./ R                           # normalised outer radii
    xa = (0.0, xb[1], xb[2], xb[3], xb[4])
    ρ̄_SI = sum(ρ_SI[i] * (xb[i]^3 - xa[i]^3) for i in 1:5)
    p_unit = (4π * G_SI / 3) * ρ̄_SI^2 * R^2

    build_model("Ganymede", [xa[i]..xb[i] for i in 1:5],
                [[ρ_SI[i] / ρ̄_SI] for i in 1:5],
                [let v = isinf(κ_SI[i]) ? Inf : κ_SI[i]/p_unit; r -> v end for i in 1:5],
                [let v = μ_SI[i]/p_unit; r -> v end for i in 1:5],
                [n_core, n_solid, n_solid, n_solid, n_solid];
                R_SI = R, ρ̄_SI = ρ̄_SI,
                # core and ocean have constant ρ (ρ₀′ = 0), so Adams-Williamson is wrong there
                aw = [μ_SI[i] != 0.0 for i in 1:5],
                Ω_SI = 2π / (7.155 * 86400))   # synchronous with the 7.155 d orbit
end
