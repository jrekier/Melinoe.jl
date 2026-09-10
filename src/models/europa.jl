# ── Europa ────────────────────────────────────────────────────────────────────
# Fe-FeS core | silicate mantle | ocean | ice shell. C/MR² = 0.3547 is low enough to
# demand a metal core (Gomez Casajus et al. 2021).
#
# Densities come from material physics; BOTH interior radii are solved from the observed
# mass and C/MR². The H2O layer is therefore an output, and the ocean is whatever is left
# between the solved mantle top and the prescribed ice shell. Two unknowns, two
# constraints, by Newton — there is no closed form once both radii are free.

"""
    Europa(; h_ice_km=20.0, ρ_core=5150.0, ρ_mantle=3300.0, n=40) -> PlanetModel

Europa: Fe-FeS core, silicate mantle, ocean `layers[3]`, ice shell. `h_ice_km` sets the
shell; the core radius, mantle radius and hence the ocean thickness all follow from the
mass and moment of inertia.

| quantity | value | source |
|---|---|---|
| mass | 4.7998e22 kg | Anderson et al. 1998, *Science* **281**, 2019 |
| mean radius | 1560.8 km | Anderson et al. 1998 |
| `C/MR²` | 0.3547 | Gomez Casajus et al. 2021, *Icarus* **358**, 114187 |
| orbital period | 3.551 d | — |

Mass and moment of inertia are matched by construction. With the defaults the core comes
out near 640 km under a 1460 km mantle, leaving an H2O layer of about 100 km — a core at
the eutectic Fe-FeS density, the lightest composition that still works. A mantle at
3800 kg/m³ needs no core at all and is rejected.

`pressure_love(m; forcing_layer = 3)` gives the ocean's pressure Love numbers.
"""
function Europa(; h_ice_km::Float64 = 20.0, ρ_core::Float64 = 5150.0,
                  ρ_mantle::Float64 = 3300.0, ρ_ocean::Float64 = 1020.0,
                  ρ_ice::Float64 = 937.0, n::Int = 40)
    R = 1560.8e3
    M_obs, Cnd = 4.7998e22, 0.3547
    ρ̄ = M_obs / (4π/3 * R^3)
    xi = 1 - h_ice_km*1e3/R                       # base of the ice shell
    (ρ_core > ρ_mantle > ρ_ocean > ρ_ice) || throw(ArgumentError(
        "Europa: densities must decrease outward, got $((ρ_core, ρ_mantle, ρ_ocean, ρ_ice))."))

    # Mass and moment of inertia as functions of the two unknown radii. Both residuals
    # are sums of the same form, so the Jacobian is exact and the solve is quadratic.
    F(xc, xm) = (ρ_core*xc^3 + ρ_mantle*(xm^3-xc^3) + ρ_ocean*(xi^3-xm^3) + ρ_ice*(1-xi^3) - ρ̄,
                 ρ_core*xc^5 + ρ_mantle*(xm^5-xc^5) + ρ_ocean*(xi^5-xm^5) + ρ_ice*(1-xi^5) - 2.5*Cnd*ρ̄)
    xc, xm = 0.4, 0.9
    for _ in 1:100
        f1, f2 = F(xc, xm)
        a, b = 3xc^2*(ρ_core-ρ_mantle), 3xm^2*(ρ_mantle-ρ_ocean)   # ∂/∂xc, ∂/∂xm of F1
        c, d = 5xc^4*(ρ_core-ρ_mantle), 5xm^4*(ρ_mantle-ρ_ocean)   #        …of F2
        δ = a*d - b*c
        abs(δ) < 1e-30 && break
        xc -= ( d*f1 - b*f2)/δ
        xm -= (-c*f1 + a*f2)/δ
        xc = clamp(xc, 1e-6, 0.999); xm = clamp(xm, 1e-6, xi - 1e-9)
        hypot(f1, f2) < 1e-10 && break
    end
    (0 < xc < xm < xi && hypot(F(xc, xm)...) < 1e-6) || throw(ArgumentError(
        "Europa: no interior with ρ_core = $ρ_core, ρ_mantle = $ρ_mantle satisfies M and " *
        "C/MR² with h_ice_km = $h_ice_km."))
    ρ = (ρ_core, ρ_mantle, ρ_ocean, ρ_ice)

    xa = (0.0, xc, xm, xi);  xb = (xc, xm, xi, 1.0)
    μ_SI = (60.0e9, 60.0e9, 0.0, 3.3e9)           # solid Fe-FeS core; ocean is fluid
    κ_SI = (130.0e9, 120.0e9, 2.2e9, 9.0e9)
    ρ̄_c  = sum(ρ[i]*(xb[i]^3 - xa[i]^3) for i in 1:4)
    p_u  = (4π*G_SI/3) * ρ̄_c^2 * R^2

    build_model("Europa", [xa[i]..xb[i] for i in 1:4], [[ρ[i]/ρ̄_c] for i in 1:4],
                [let v = κ_SI[i]/p_u; r -> v end for i in 1:4],
                [let v = μ_SI[i]/p_u; r -> v end for i in 1:4], fill(n, 4);
                R_SI = R, ρ̄_SI = ρ̄_c,
                # constant ρ per layer ⇒ ρ₀′ = 0, so Adams-Williamson is wrong in the ocean
                aw = [μ_SI[i] != 0.0 for i in 1:4],
                Ω_SI = 2π / (3.551 * 86400))
end
