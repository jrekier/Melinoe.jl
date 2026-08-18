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

`C_nd` is less certain than its formal error: it comes from Darwin–Radau, which
non-hydrostatic stress can inflate by over 10% (Gao & Stevenson 2013, *Icarus*
**226**, 1185), and the Juno reanalysis points to a larger true value (Gomez Casajus
et al. 2022, *GRL* **49**, e2022GL099475).

`pressure_love(m; forcing_layer = 4)` gives the ocean's pressure Love numbers.
"""
function Ganymede(; h_ice_km::Float64 = 100.0, n_core::Int = 50, n_solid::Int = 40)
    R   = 2631.2e3                      # mean radius, Baland & Van Hoolst 2010
    Rc  = 720.0e3; Rm = 1820.0e3; Rhp = 2284.0e3
    Ro  = R - h_ice_km*1e3

    M_obs, Cnd_obs = 1.482e23, 0.3105        # Anderson et al. 1996, Nature 384, 541
    edges = (0.0, Rc, Rm, Rhp, Ro, R)
    # hydrosphere densities held fixed; core and mantle scaled to the observables
    ρ_hyd = (1346.0, 1100.0, 937.0)     # HP ice, ocean, shell
    ρ_ref = (5777.9, 3291.5)            # core, mantle — starting profile

    v3(i) = 4π/3 * (edges[i+1]^3 - edges[i]^3)      # mass weight
    v5(i) = 8π/15 * (edges[i+1]^5 - edges[i]^5)     # polar-MoI weight
    rest3 = sum(v3(i)*ρ_hyd[i-2] for i in 3:5)
    rest5 = sum(v5(i)*ρ_hyd[i-2] for i in 3:5)
    α, β = [v3(1)*ρ_ref[1]  v3(2)*ρ_ref[2]
            v5(1)*ρ_ref[1]  v5(2)*ρ_ref[2]] \ [M_obs - rest3, Cnd_obs*M_obs*R^2 - rest5]

    ρ_SI = (α*ρ_ref[1], β*ρ_ref[2], ρ_hyd...)                # ≈ 5662, 3313, …
    μ_SI = (0.0, 50.0e9, 6.6e9, 0.0, 3.3e9)                  # Hussmann et al. 2016, CMDA 126, 131
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
