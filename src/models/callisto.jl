# ── Callisto ──────────────────────────────────────────────────────────────────
# Rock-ice interior | ocean | ice shell. C/MR² = 0.3549 is the highest of the Galilean
# satellites and too high for full differentiation (Anderson et al. 2001): the interior
# keeps its ice, so it is modelled as one rock-ice mixture rather than core + mantle.
#
# The mixture DENSITY and its RADIUS are both solved from the observed mass and C/MR².
# Both constraints are linear in that density at fixed radius, so eliminating it leaves
# x² = B/A in closed form — the same algebra as `Mercury`.

"""
    Callisto(; h_ice_km=100.0, ρ_ocean=1100.0, ρ_ice=937.0, n=40) -> PlanetModel

Callisto: undifferentiated rock-ice interior, ocean `layers[2]`, ice shell. `h_ice_km`
sets the shell; the ocean thickness follows from the mass and moment of inertia.

| quantity | value | source |
|---|---|---|
| `GM` | 7179.292 km³ s⁻² | Anderson et al. 2001, *Icarus* **153**, 157 |
| mean radius | 2410.3 km | Anderson et al. 2001 |
| `C/MR²` | 0.3549 ± 0.0042 | Anderson et al. 2001 |
| orbital period | 16.689 d | — |

The solved interior density (~1980 kg/m³ with the defaults) sits between ice and rock,
which is the point: Callisto never fully separated. Mass and moment of inertia are matched
by construction; the Love numbers are predictions.
"""
function Callisto(; h_ice_km::Float64 = 100.0, ρ_ocean::Float64 = 1100.0,
                    ρ_ice::Float64 = 937.0, n::Int = 40)
    R  = 2410.3e3
    GM = 7179.292e9                                # m³/s², Anderson et al. 2001
    M_obs = GM / G_SI
    Cnd = 0.3549
    ρ̄  = M_obs / (4π/3 * R^3)
    xi = 1 - h_ice_km*1e3/R                        # base of the ice shell

    # both constraints are linear in ρ_int at fixed x, so x² = B/A closes in one step
    A = ρ̄          - ρ_ocean*xi^3 - ρ_ice*(1 - xi^3)     # = x³(ρ_int − ρ_ocean)
    B = 2.5*Cnd*ρ̄  - ρ_ocean*xi^5 - ρ_ice*(1 - xi^5)     # = x⁵(ρ_int − ρ_ocean)
    (A > 0 && B > 0) || throw(ArgumentError(
        "Callisto: no interior denser than the ocean satisfies M and C/MR² with " *
        "h_ice_km = $h_ice_km."))
    x = sqrt(B/A)
    ρ_int = ρ_ocean + A/x^3
    x < xi || throw(ArgumentError(
        "Callisto: h_ice_km = $h_ice_km leaves no room for an ocean (interior reaches " *
        "$(round(x*R/1e3, digits=1)) km, ice base $(round(xi*R/1e3, digits=1)) km)."))

    xa = (0.0, x, xi);  xb = (x, xi, 1.0)
    ρ    = (ρ_int, ρ_ocean, ρ_ice)
    μ_SI = (30.0e9, 0.0, 3.3e9)                    # rock-ice mixture; ocean is fluid
    κ_SI = (60.0e9, 2.2e9, 9.0e9)
    ρ̄_c  = sum(ρ[i]*(xb[i]^3 - xa[i]^3) for i in 1:3)
    p_u  = (4π*G_SI/3) * ρ̄_c^2 * R^2

    build_model("Callisto", [xa[i]..xb[i] for i in 1:3], [[ρ[i]/ρ̄_c] for i in 1:3],
                [let v = κ_SI[i]/p_u; r -> v end for i in 1:3],
                [let v = μ_SI[i]/p_u; r -> v end for i in 1:3], fill(n, 3);
                R_SI = R, ρ̄_SI = ρ̄_c,
                aw = [μ_SI[i] != 0.0 for i in 1:3],
                Ω_SI = 2π / (16.689 * 86400))
end
