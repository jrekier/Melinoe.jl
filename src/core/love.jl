# ── Static / dynamic tidal solve and Love numbers ─────────────────────────────
#
# Solve (A − ω²B)·x = f for an applied external potential (default Φ_T = r^ℓ, unit
# amplitude at the surface). Derived step by step in examples/1_love_numbers_tutorial.ipynb.
#
# Sign convention: the code carries gravity as −g (inward), so δφ is the
# self-gravity perturbation and the geophysics Love numbers are h = U(1) ≤ 0,
# l = V(1) ≤ 0 (report |h|, |l|); k = δφ(1) already has the correct positive sign.

"""
    forced_solve(Amat, Bmat, ops, ns; ℓ, potential=(r->r^ℓ), amplitude=1.0,
                 layer_mask=trues(length(ops)), ω²=0.0) -> x

Solve `(A − ω²B)x = f` for the applied potential `amplitude·potential(r)` on the
flagged layers, returning the full spectral vector `x = [U; V/P; δφ]`.
`ω²=0` is the static (tidal) response.
"""
function forced_solve(Amat, Bmat, ops, ns; ℓ::Int, potential = (r -> r^ℓ),
                      amplitude::Real = 1.0, layer_mask = trues(length(ops)),
                      ω² = 0.0, jrhs = nothing)
    f = potential_forcing(ops, ns; ℓ, potential, amplitude, layer_mask, jrhs)
    return (Amat - ω² .* Bmat) \ f
end

# Static + fluid surface: neither inertia nor shear traction pins the tangential
# displacement, so the reconstructed V(R) is noise. `l` is reported as NaN.
_static_fluid_surface(ops, ω²) = ops[end].fluid && iszero(ω²)

# Read the surface Love numbers (h, l, k) off a solved response vector.
function read_love(x, layers, ops, ns, ℓ; ω² = 0.0)
    L    = length(layers)
    Ntot = sum(ns)
    cumN = cumsum([0; ns])
    SL   = ops[L].S
    r_s  = rightendpoint(layers[L].domain)

    xU  = x[cumN[L]+1       : cumN[L+1]]
    xVP = x[Ntot+cumN[L]+1  : Ntot+cumN[L+1]]
    xδφ = x[2Ntot+cumN[L]+1 : 2Ntot+cumN[L+1]]

    h = Fun(SL, xU)(r_s)
    k = Fun(SL, xδφ)(r_s)
    l = if _static_fluid_surface(ops, ω²)
            NaN
        elseif ops[L].fluid
            DU = Fun(SL, xU)'
            P  = Fun(SL, xVP)
            Pκ = is_incompressible(layers[L]) ? 0.0 : P(r_s) / layers[L].κ(r_s)
            r_s / (ℓ*(1+ℓ)) * (DU(r_s) + 2h/r_s + Pκ)
        else
            Fun(SL, xVP)(r_s)
        end
    return (; h, l, k)
end

"""
    love_numbers(layers, ℓ; ω²=0.0)

Static (ω=0) or dynamic tidal Love numbers `(; h, l, k)` at degree `ℓ`.

`l` is `NaN` for a static solve on a fluid-surfaced body (`Kelvin()`,
`PREM(ocean=true)`), where it is undetermined; `h` and `k` are unaffected.
Pass a nonzero `ω²` for a meaningful `l`.
"""
function love_numbers(layers::Vector{Layer}, ℓ::Int; ω² = 0.0)
    Amat, Bmat, ops, ns, jrhs = assemble_planet(layers, ℓ)
    x = forced_solve(Amat, Bmat, ops, ns; ℓ, ω², jrhs)
    read_love(x, layers, ops, ns, ℓ; ω²)
end
