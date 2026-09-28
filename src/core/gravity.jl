# ── Background gravity & potential from a density profile ─────────────────────
# `self_gravity` and `background_potential` derive g₀(r) and φ₀(r) from a density
# given either as polynomial coefficient vectors (ρ(r)=Σⱼ c[j]·r^(j-1), integrated
# exactly by the power rule) or as callables, which are first fit to a polynomial.

"Exact ∫_{r1}^{r2} ρ(r) r² dr for ρ with coefficients `c`."
poly_r2_integral(c, r1, r2) =
    sum(c[j] * (r2^(j+2) - r1^(j+2)) / (j+2) for j in eachindex(c))

"Exact ∫_{r1}^{r2} ρ(r) r⁴ dr for ρ with coefficients `c` (equatorial-MoI kernel)."
poly_r4_integral(c, r1, r2) =
    sum(c[j] * (r2^(j+4) - r1^(j+4)) / (j+4) for j in eachindex(c))

# Least-squares fit of each density function to a degree-`degree` monomial
# polynomial, one coefficient vector per layer. Front-end for the `Function`
# methods below.
_fit_polys(ρs, domains, degree) = map(ρs, domains) do ρ, d
    rs = range(leftendpoint(d), rightendpoint(d), length = 4*(degree+1))
    [rs[i]^j for i in eachindex(rs), j in 0:degree] \ ρ.(rs)
end

"""
    self_gravity(coeffs, domains) -> Vector{Function}

Per-layer background gravity `g₀(r) = 3 M(r) / r²` in normalised units, computed
analytically from the mean-normalised density polynomials `coeffs[i]` on
`domains[i]`. Returns one `g₀` callable per layer. Mean normalisation
(`3·Σᵢ∫ρᵢr²dr = 1`) gives `g(1) = 1`. A constant-density layer is `coeffs[i] = [ρᵢ]`.

`build_model` calls this to fill each `Layer`'s `g₀`.
"""
function self_gravity(coeffs::Vector{<:Vector}, domains)
    L = length(domains)
    M_inner = zeros(L)                     # enclosed ∫ρr²dr up to each layer's inner edge
    for i in 2:L
        a = leftendpoint(domains[i-1]); b = rightendpoint(domains[i-1])
        M_inner[i] = M_inner[i-1] + poly_r2_integral(coeffs[i-1], a, b)
    end
    return Function[
        let ai = leftendpoint(domains[i]), Mi = M_inner[i], ci = coeffs[i]
            r -> r == 0.0 ? 0.0 : 3 * (Mi + poly_r2_integral(ci, ai, r)) / r^2
        end
        for i in 1:L
    ]
end

"""
    self_gravity(ρs::Vector{<:Function}, domains; degree=12) -> Vector{Function}

Density **functions** `ρs[i]`, one per layer, in place of coefficient vectors. Each
is sampled across its layer and least-squares fit to a degree-`degree` polynomial,
then passed to the method above. Raise `degree` for a strongly varying density.
"""
self_gravity(ρs::Vector{<:Function}, domains; degree::Int = 12) =
    self_gravity(_fit_polys(ρs, domains, degree), domains)

# ── Background potential (reconstruction / diagnostics; not used by the elastic solve) ─

"""
    background_potential(coeffs, domains) -> Vector{Function}

Per-layer background potential `φ₀(r)` from the density polynomials `coeffs[i]`,
exact, in the `φ₀(∞)=0` gauge. By the shell theorem `g₀` needs only the interior
mass, but `φ₀` is non-local: each layer carries an additional far-field `B/r` term
and a continuity constant `C` matched inward from the surface.
"""
function background_potential(coeffs::Vector{<:Vector}, domains)
    L = length(domains)
    a = [leftendpoint(d)  for d in domains]
    b = [rightendpoint(d) for d in domains]

    poly_φ(c, r) = sum(c[j+1] * 3*r^(j+2) / ((j+3)*(j+2)) for j in 0:length(c)-1)  # this layer's own φ

    # forward sweep: mass interior to each layer's inner edge (as in self_gravity)
    m_left = zeros(L)
    for i in 2:L
        m_left[i] = m_left[i-1] + poly_r2_integral(coeffs[i-1], a[i-1], b[i-1])
    end
    M_total = m_left[L] + poly_r2_integral(coeffs[L], a[L], b[L])

    # far-field 1/r coefficient, then continuity constants matched inward from the surface
    B = [3 * (poly_r2_integral(coeffs[i], 0.0, a[i]) - m_left[i]) for i in 1:L]
    C = zeros(L)
    C[L] = -3*M_total - B[L]/b[L] - poly_φ(coeffs[L], b[L])
    for i in L-1:-1:1
        r0 = b[i]
        C[i] = B[i+1]/r0 + poly_φ(coeffs[i+1], r0) + C[i+1] -
               B[i]/r0   - poly_φ(coeffs[i],   r0)
    end

    return Function[ let Bi=B[i], ci=coeffs[i], Ci=C[i]
        r -> (Bi == 0.0 ? 0.0 : Bi/r) + poly_φ(ci, r) + Ci
    end for i in 1:L ]
end

"""
    background_potential(ρs::Vector{<:Function}, domains; degree=12) -> Vector{Function}

Density-function form: each `ρs[i]` is fit to a degree-`degree` polynomial, as in
the `self_gravity` function method, then passed to the polynomial method above.
"""
background_potential(ρs::Vector{<:Function}, domains; degree::Int = 12) =
    background_potential(_fit_polys(ρs, domains, degree), domains)
