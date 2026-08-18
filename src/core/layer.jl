# ── Layer definition ─────────────────────────────────────────────────────────
#
# One spherical shell of the planet, on `domain = a..b`, discretised with `n`
# spectral modes. `κ, μ, ρ₀, g₀` are callables of radius `r` (normalised units):
#   κ  — bulk modulus  (Inf ⇒ incompressible)
#   μ  — shear modulus (0   ⇒ fluid)
#   ρ₀ — background density
#   g₀ — background gravity g₀(r) = dφ₀/dr, supplied analytically (see
#        `self_gravity`), which avoids the 1/r² singularity that appears if it
#        is differentiated from a Chebyshev-expanded potential.

@kwdef struct Layer
    domain
    n   :: Int
    κ   :: Function
    μ   :: Function
    ρ₀  :: Function
    g₀  :: Function
    aw  :: Bool = true   # fluid Poisson closure. true: Adams-Williamson, ρ₀′=−ρ₀²g/κ
                         # (adiabatic shell). false: keep the actual ρ₀′ — needed for a
                         # core reaching the centre, or a subadiabatic tabulated κ.
    dahlen :: Bool = false  # fluid static closure (Dahlen 1974); see `dahlenize`.
                            # Static solves only. Ignored for solids.
    ν_SI :: Float64 = 0.0   # kinematic viscosity [m²/s]; 0 ⇒ inviscid. Unused by the
                            # gravito-elastic solvers.
end

is_fluid(l::Layer) = l.μ((leftendpoint(l.domain) + rightendpoint(l.domain)) / 2) == 0.0
is_incompressible(l::Layer) = isinf(l.κ((leftendpoint(l.domain) + rightendpoint(l.domain)) / 2))
