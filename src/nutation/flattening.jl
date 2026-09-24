# ── Dynamical flattening from Clairaut's equation ─────────────────────────────
#
# Hydrostatic flattening ε(r) of the equipotential (level) surfaces of a rotating
# body, from the density profile via Clairaut's equation in Radau form:
#
#   D(r) = ρ(r)/⟨ρ⟩(r) = ρ₀(r)·r/g₀(r)                (⟨ρ⟩ = 3M/r³ = g/r)
#   r dη/dr = 6 − 6D(1+η) + η − η² ,   η(0)=0 ,   η = r ε′/ε
#   surface figure:  ε(a) = (5m/2)/(2+η(a)) ,   m = Ω̂² = (Ω/ω₀)²
#
# ε and η are continuous across density jumps (D jumps; level surfaces stay
# smooth), so the RK4 integration runs straight through the interfaces. Validated
# on PREM: ε(a)=1/300.2 (Earth hydrostatic 1/299.8), ε(CMB)=1/393, ε(ICB)=1/414.

"""
    flattening(model; Ω_SI=Ω_EARTH, ngrid=40000)
        -> (; ε, η, εa, ηa, rs, εs, ηs)

Solve Clairaut's equation for the dynamical flattening. `ε(r)` and `η(r)` are
callables (nearest-grid); `εa=ε(1)`, `ηa=η(1)` are the surface values.
"""
function flattening(model::PlanetModel; Ω_SI = nothing, ngrid::Int = 40000)
    Ω̂² = (rotation_rate(model, Ω_SI) / model.ω_unit)^2
    layers = model.layers
    function ρg(r)
        for l in layers
            r <= rightendpoint(l.domain) + 1e-12 && return l.ρ₀(r), l.g₀(r)
        end
        return layers[end].ρ₀(r), layers[end].g₀(r)
    end
    Dfun(r) = r < 1e-10 ? 1.0 : (rg = ρg(r); rg[1]*r/rg[2])
    f(r, η) = (6 - 6*Dfun(r)*(1+η) + η - η^2) / r

    r0 = 1e-5; h = (1.0 - r0)/ngrid
    r = r0; η = 0.0; L = 0.0
    rs = Float64[r]; ηs = Float64[η]; Ls = Float64[L]
    for _ in 1:ngrid
        k1=f(r,η); k2=f(r+h/2,η+h/2*k1); k3=f(r+h/2,η+h/2*k2); k4=f(r+h,η+h*k3)
        m1=η/r; m2=(η+h/2*k1)/(r+h/2); m3=(η+h/2*k2)/(r+h/2); m4=(η+h*k3)/(r+h)
        η += h/6*(k1+2k2+2k3+k4); L += h/6*(m1+2m2+2m3+m4); r += h
        push!(rs,r); push!(ηs,η); push!(Ls,L)
    end
    ηa = ηs[end]; εa = (5Ω̂²/2)/(2 + ηa)
    εs = εa .* exp.(Ls .- Ls[end])
    idx(rq) = clamp(searchsortedfirst(rs, rq), 1, length(rs))
    return (; ε = rq -> εs[idx(rq)], η = rq -> ηs[idx(rq)], εa, ηa, rs, εs, ηs)
end

# mean moment A=(8π/3)∫ρr⁴dr and dynamical ellipticity e=(C−A)/A over a radial range
function _region_moment_e(m, flt, x1, x2)
    R = m.R_SI; ρ̄ = m.ρ̄_SI
    ρ̂(x) = (for l in m.layers; x ≤ rightendpoint(l.domain)+1e-12 && return l.ρ₀(x); end; m.layers[end].ρ₀(x))
    rs, εs, ηs = flt.rs, flt.εs, flt.ηs
    j = (x1-1e-12 .≤ rs .≤ x2+1e-12); r, ε, η = rs[j], εs[j], ηs[j]; ρ = ρ̂.(r)
    trap(y) = sum((y[1:end-1] .+ y[2:end]) ./ 2 .* diff(r))
    A = (8π/3)*ρ̄*R^5*trap(ρ.*r.^4); CmA = (8π/15)*ρ̄*R^5*trap(ρ.*r.^4 .*ε.*(5 .+η))
    return A, CmA/A
end
