# ── Kelvin sphere ─────────────────────────────────────────────────────────────
# Uniform-density self-gravitating sphere; the analytic Love-number benchmark.
# μ̃ = 19μ/(2ρgR), so a solid shell takes normalised μ = 2μ̃/19 (ρ = g(1) = R = 1).

"""
    Kelvin(; μ̃=0.0, κ̃=Inf, n=60) -> PlanetModel

Uniform sphere, ρ=1, g(r)=r. `μ̃=0` gives the incompressible fluid Kelvin Earth;
`μ̃>0` a solid shell of shear modulus `2μ̃/19` and finite bulk modulus `κ̃` (an
incompressible solid is unsupported).

Analytic: `h₂ = (5/2)/(1+μ̃)`, `k₂ = (3/2)/(1+μ̃)`, and an ℓ=2 f-mode at `ω² = 4/5`.

`l₂` depends on how you reach the fluid sphere, since the limits do not commute —
an inviscid fluid has no shear-traction condition, a μ̃ → 0 solid keeps one:

| route | pinned by | l₂ |
|---|---|---|
| solid, μ̃ → 0 | shear traction | `(3/4)/(1+μ̃)` → 3/4 |
| fluid, ω² → 0⁺ | inertia | 5/4 |

With `μ̃ = 0` and `ω² = 0` neither applies, and `love_numbers` returns `l = NaN`.
"""
function Kelvin(; μ̃::Float64 = 0.0, κ̃ = Inf, n::Int = 60)
    if μ̃ == 0.0
        κfun = r -> Inf; μfun = r -> 0.0
    else
        @assert isfinite(κ̃) "solid Kelvin needs a finite κ̃ (incompressible solid unsupported)"
        μval = 2μ̃/19
        κfun = r -> κ̃;   μfun = r -> μval
    end
    build_model("Kelvin", [0.0..1.0], [[1.0]], [κfun], [μfun], [n];
                R_SI = 6371.0e3, ρ̄_SI = 5515.0)
end
