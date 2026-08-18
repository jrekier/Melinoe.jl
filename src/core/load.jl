# ── Tabulated-model I/O ───────────────────────────────────────────────────────
#
# Reads the usual published layout: one row per sampled radius, a header naming the
# columns, optionally with `[unit="…"]` annotations. `examples/PREM_1s.csv` is a
# worked example (real PREM, as distributed).
#
# Columns are matched by name, case-insensitively, and anything unrecognised (Q_μ,
# Q_κ, …) is ignored:
#
#   position   radius | depth                            (one of the two)
#   density    density | rho
#   elasticity K | kappa | bulk   and   mu | shear        (moduli, preferred), or
#              vp,  vs                                    (isotropic velocities), or
#              vpv, vph, vsv, vsh [, eta]                 (transversely isotropic)
#   layer      layer | region | name                      (optional, see below)
#
# Layers: with a `layer` column, rows are grouped by it. Without one — the usual
# published convention — a REPEATED radius marks a discontinuity and starts a new
# layer, which is how PREM delimits its regions.

const _CSV_ALIASES = Dict{Symbol,Vector{String}}(
    :layer => ["layer", "region", "name"],
    :r     => ["radius", "r", "r_m", "r_km"],
    :depth => ["depth", "z"],
    :ρ     => ["density", "rho"],
    :vp    => ["vp", "vp_iso"],  :vs  => ["vs", "vs_iso"],
    :vpv   => ["vpv"], :vph => ["vph"], :vsv => ["vsv"], :vsh => ["vsh"],
    :η     => ["eta"],
    :κ     => ["k", "kappa", "bulk", "bulk_modulus"],
    :μ     => ["mu", "shear", "shear_modulus"],
)

_strip_unit(h) = lowercase(strip(replace(h, r"\[.*?\]" => ""), ['"', '\'', ' ', '\t']))
_unit_of(h) = (m = match(r"unit\s*=\s*\"?([^\"\]]*)", h);
               m === nothing ? "" : replace(lowercase(strip(m.captures[1])), " " => ""))

# Scale a declared unit to SI. An empty unit means the column is already SI, so an
# unannotated km column would read 1000× small — caught by the sanity check on R.
function _unit_scale(kind::Symbol, u::AbstractString, col::AbstractString)
    tbl = kind === :length   ? Dict(""=>1.0, "m"=>1.0, "km"=>1e3) :
          kind === :density  ? Dict(""=>1.0, "kg/m^3"=>1.0, "kg/m3"=>1.0,
                                    "g/cm^3"=>1e3, "g/cm3"=>1e3, "g/cc"=>1e3) :
          kind === :velocity ? Dict(""=>1.0, "m/s"=>1.0, "km/s"=>1e3) :
                               Dict(""=>1.0, "pa"=>1.0, "mpa"=>1e6, "gpa"=>1e9)
    haskey(tbl, u) || throw(ArgumentError(
        "column \"$col\": unit \"$u\" not recognised for a $kind (expected one of " *
        join(filter(!isempty, sort(collect(keys(tbl)))), ", ") * ")"))
    return tbl[u]
end

# Parse a field, tolerating quoting and Mathematica's a*^b exponent form.
_unquote(s) = strip(s, ['"', '\'', ' ', '\t'])
_numstr(s)  = replace(_unquote(s), "*^" => "e")

"""
Read a CSV into `(; cols, meta)`: `cols` maps a canonical symbol to its SI-scaled
column (`:layer` stays a `Vector{String}`), `meta` holds `key=value` pairs found in
leading `#` comments. Leading `#`, `%` or `(` lines are skipped.
"""
function _read_csv_table(path)
    raw = [l for l in readlines(path) if !isempty(strip(l))]
    isempty(raw) && throw(ArgumentError("$path: file is empty"))

    meta = Dict{String,String}()
    for l in raw
        s = strip(l)
        startswith(s, ('#', '%', '(')) || break
        for m in eachmatch(r"(\w+)\s*=\s*([^\s,]+)", s); meta[lowercase(m[1])] = m[2]; end
    end
    i0 = findfirst(l -> !startswith(strip(l), ('#', '%', '(')), raw)
    i0 === nothing && throw(ArgumentError("$path: no data rows, only comments"))

    header = strip.(split(raw[i0], ','))
    all(f -> tryparse(Float64, _numstr(f)) !== nothing, header) && throw(ArgumentError(
        "$path: line $i0 parses as data, so the file appears to have no header. " *
        "A header naming the columns is required (e.g. \"radius,density,vp,vs\")."))

    names = _strip_unit.(header)
    units = _unit_of.(header)
    idx = Dict{Symbol,Int}()
    for (sym, aliases) in _CSV_ALIASES
        j = findfirst(in(aliases), names)
        j !== nothing && (idx[sym] = j)
    end

    body = raw[i0+1:end]
    isempty(body) && throw(ArgumentError("$path: header present but no data rows"))
    ncol = length(header)
    rows = Vector{Vector{SubString{String}}}(undef, length(body))
    for (k, l) in enumerate(body)
        f = split(strip(l), ',')
        length(f) == ncol || throw(ArgumentError(
            "$path line $(i0+k): $(length(f)) fields, expected $ncol"))
        rows[k] = f
    end

    kind_of(s) = s in (:r, :depth) ? :length : s === :ρ ? :density :
                 s in (:vp, :vs, :vpv, :vph, :vsv, :vsh) ? :velocity :
                 s in (:κ, :μ) ? :modulus : :none
    cols = Dict{Symbol,Any}()
    for (sym, j) in idx
        if sym === :layer
            cols[sym] = [String(_unquote(r[j])) for r in rows]
            continue
        end
        kind = kind_of(sym)
        # +Inf is meaningful in a modulus or velocity column — it is how an
        # incompressible layer is written — but nowhere else.
        inf_ok = kind in (:modulus, :velocity)
        v = Vector{Float64}(undef, length(rows))
        for (k, r) in enumerate(rows)
            x = tryparse(Float64, _numstr(r[j]))
            x === nothing && throw(ArgumentError(
                "$path line $(i0+k), column \"$(header[j])\": cannot parse $(repr(String(r[j]))) as a number"))
            (isfinite(x) || (inf_ok && x == Inf)) || throw(ArgumentError(
                "$path line $(i0+k), column \"$(header[j])\": value is $x, which no " *
                "layer profile can be fitted through"))
            v[k] = x
        end
        cols[sym] = kind === :none ? v : v .* _unit_scale(kind, units[j], header[j])
    end
    return (; cols, meta)
end

# Elastic moduli (Pa) from whatever the file supplied. Velocities are reduced to the
# Voigt isotropic equivalent, which for A=ρVph², C=ρVpv², N=ρVsh², L=ρVsv²,
# F=η(A−2L) is  κ = (C+4A−4N+4F)/9,  μ = (C+A+6L+5N−2F)/15  — exactly ρVp²−4ρVs²/3
# and ρVs² when the medium is isotropic.
function _moduli_from(cols, ρ, path)
    if haskey(cols, :κ) && haskey(cols, :μ)
        return cols[:κ], cols[:μ]
    end
    iso  = haskey(cols, :vp)  && haskey(cols, :vs)
    anis = haskey(cols, :vpv) && haskey(cols, :vph) && haskey(cols, :vsv) && haskey(cols, :vsh)
    (iso || anis) || throw(ArgumentError(
        "$path: need either moduli (K and mu) or velocities (vp and vs, or the four " *
        "vpv/vph/vsv/vsh) to build the elastic model"))
    vpv = anis ? cols[:vpv] : cols[:vp];  vph = anis ? cols[:vph] : cols[:vp]
    vsv = anis ? cols[:vsv] : cols[:vs];  vsh = anis ? cols[:vsh] : cols[:vs]
    η   = get(cols, :η, ones(length(ρ)))
    A = ρ .* vph.^2;  C = ρ .* vpv.^2;  N = ρ .* vsh.^2;  L = ρ .* vsv.^2
    F = η .* (A .- 2 .* L)
    κ = (C .+ 4A .- 4N .+ 4F) ./ 9
    μ = (C .+ A .+ 6L .+ 5N .- 2F) ./ 15
    # An infinite vp marks an incompressible layer. The Voigt shear combination then
    # degenerates to Inf − Inf, so take μ from the shear velocities directly.
    for k in eachindex(κ)
        isfinite(A[k]) && isfinite(C[k]) && continue
        κ[k] = Inf
        μ[k] = ρ[k] * vsh[k]^2
    end
    return κ, μ
end

# Lowest-degree least-squares fit reaching `rtol`. A degree whose coefficients blow
# past 100× the data scale is skipped as ill-conditioned; if nothing resolves the
# layer the BEST fit is returned, not the last, together with its relative error.
function _lsq_fit(ts, ys, maxdeg; rtol = 1e-3)
    scale = maximum(abs, ys) + eps()
    best_c, best_err = [sum(ys)/length(ys)], Inf
    for d in 0:min(maxdeg, length(ts) - 1)
        c = ([t^j for t in ts, j in 0:d]) \ ys
        err = sqrt(sum(abs2, [evalpoly(t, c) for t in ts] .- ys)/length(ys))/scale
        err < rtol && return c, err
        maximum(abs, c) > 100*scale && continue
        err < best_err && ((best_c, best_err) = (c, err))
    end
    return best_c, best_err
end

"""
    write_planet_csv(model, path; n=40, format=:moduli) -> path

Sample a `PlanetModel` onto `n` radii per layer and write it as CSV. Both layouts
`load_planet_csv` accepts can be written, and both round-trip:

- `:moduli` — `layer, radius [m], density [kg/m³], K [Pa], mu [Pa]`. An explicit
  layer column, SI throughout, moduli exactly as the model stores them.
- `:velocity` — `radius [km], depth [km], density [g/cm³], vp [km/s], vs [km/s]`,
  the layout tabulated models use: no layer column, boundaries marked by repeated
  radii, ordered surface inward. Compare `examples/PREM_1s.csv`.

Rotation is carried in a header comment either way. An incompressible layer
(`κ = Inf`) is written as a literal `Inf` in the `K` column, or as an infinite `vp`,
and `load_planet_csv` reads it back as incompressible.
"""
function write_planet_csv(model::PlanetModel, path; n::Int = 40, format::Symbol = :moduli)
    format in (:moduli, :velocity) || throw(ArgumentError(
        "format must be :moduli or :velocity, got :$format"))
    R = model.R_SI; ρ̄ = model.ρ̄_SI; p_u = model.p_unit
    rows = NamedTuple[]
    for (i, l) in enumerate(model.layers), x in range(leftendpoint(l.domain),
                                                      rightendpoint(l.domain), length = n)
        push!(rows, (i = i, r = x*R, ρ = l.ρ₀(x)*ρ̄, κ = l.κ(x)*p_u, μ = l.μ(x)*p_u))
    end

    open(path, "w") do io
        @printf(io, "# Melinoe planet model  name=%s  Omega_SI=%.10g\n", model.name, model.Ω_SI)
        if format === :moduli
            println(io, "layer,radius[unit=\"m\"],density[unit=\"kg/m^3\"],K[unit=\"Pa\"],mu[unit=\"Pa\"]")
            for s in rows
                println(io, join(("layer$(s.i)", s.r, s.ρ, s.κ, s.μ), ","))
            end
        else
            println(io, "radius[unit=\"km\"],depth[unit=\"km\"],density[unit=\"g/cm^3\"]," *
                        "vp[unit=\"km/s\"],vs[unit=\"km/s\"]")
            for s in Iterators.reverse(rows)      # surface inward, as published tables run
                @printf(io, "%.6f,%.6f,%.8f,%.8f,%.8f\n", s.r/1e3, (R - s.r)/1e3,
                        s.ρ/1e3, sqrt((s.κ + 4s.μ/3)/s.ρ)/1e3, sqrt(s.μ/s.ρ)/1e3)
            end
        end
    end
    return path
end

"""
    load_planet_csv(path; name, poly_degree=5, μ_tol=1e7, Ω_SI=nothing,
                    modulus_scale=1.0, rtol=1e-3) -> PlanetModel

Build a `PlanetModel` from a tabulated interior model. Columns are matched by name
(see the top of this file); a position column (`radius` or `depth`), `density`, and
either moduli (`K`, `mu`) or velocities (`vp`/`vs`, or `vpv`/`vph`/`vsv`/`vsh` with
optional `eta`) are required. Everything else — `Q-mu`, `Q-kappa`, … — is ignored.
`[unit="…"]` annotations in the header are honoured: km, g/cm³, km/s, GPa and their
SI equivalents. Rows may run inward or outward.

Layers come from a `layer` column if there is one, else from repeated radii, the
convention published models use to mark a discontinuity. A layer whose `|μ|` never
exceeds `μ_tol` is treated as inviscid fluid and given the non-AW closure.

Per-layer profiles are least-squares polynomials of degree ≤ `poly_degree`, and any
layer that cannot be resolved to relative RMS `rtol` is reported as a warning rather
than fitted silently. `Ω_SI` sets the rotation rate, defaulting to a value recorded
in a header comment (as `write_planet_csv` emits) and otherwise to 0.

```julia
m = load_planet_csv(joinpath(pkgdir(Melinoe), "examples", "PREM_1s.csv"))
```
"""
function load_planet_csv(path; name = splitext(basename(path))[1], poly_degree::Int = 5,
                         modulus_scale = 1.0, μ_tol = 1e7, Ω_SI = nothing, rtol = 1e-3)
    cols, meta = _read_csv_table(path)

    haskey(cols, :ρ) || throw(ArgumentError("$path: no density column (density | rho)"))
    ρv = cols[:ρ]
    Kv, μv = _moduli_from(cols, ρv, path)
    Kv = Kv .* modulus_scale;  μv = μv .* modulus_scale

    # position: prefer radius; from depth alone the deepest sample is taken as the
    # centre, so r = max(depth) − depth (the reach-the-centre check below confirms it)
    r_m = haskey(cols, :r)     ? cols[:r] :
          haskey(cols, :depth) ? maximum(cols[:depth]) .- cols[:depth] :
          throw(ArgumentError("$path: no position column (radius | depth)"))

    # order inward→outward, preserving the file's ordering of repeated radii
    if issorted(r_m, rev = true)
        perm = reverse(eachindex(r_m))
    elseif issorted(r_m)
        perm = collect(eachindex(r_m))
    else
        perm = sortperm(r_m; alg = MergeSort)
    end
    r_m = r_m[perm];  ρv = ρv[perm];  Kv = Kv[perm];  μv = μv[perm]
    lname = haskey(cols, :layer) ? cols[:layer][perm] : nothing

    R = maximum(r_m)
    R > 1e4 || throw(ArgumentError(
        "$path: outermost radius is $(R) m. If the file is in km, annotate the column " *
        "as radius[unit=\"km\"]."))
    r0 = minimum(r_m)
    r0 ≤ 1e-4 * R || throw(ArgumentError(
        "$path: innermost sample is at r = $(round(r0/1e3, digits=1)) km, so the model " *
        "has a hole at the centre. The solver imposes regularity at r=0 and needs the " *
        "table to reach it."))
    x = r_m ./ R

    # ── group into layers ────────────────────────────────────────────────────
    if lname !== nothing
        order = String[];  for nm in lname; nm in order || push!(order, nm); end
        sort!(order, by = nm -> sum(x[lname .== nm]) / count(==(nm), lname))
        groups = [findall(==(nm), lname) for nm in order]
        for i in firstindex(groups):lastindex(groups)-1    # contiguity is not guaranteed here
            top, bot = maximum(x[groups[i]]), minimum(x[groups[i+1]])
            isapprox(top, bot; atol = 1e-9) || throw(ArgumentError(
                "$path: layers \"$(order[i])\" and \"$(order[i+1])\" are not contiguous — " *
                "$(round(top*R/1e3, digits=2)) km vs $(round(bot*R/1e3, digits=2)) km. " *
                "Gaps and overlaps are not resolvable into a layered model."))
        end
    else
        starts = [1]
        for i in 2:length(x)
            x[i] == x[i-1] && push!(starts, i)      # repeated radius ⇒ discontinuity
        end
        push!(starts, length(x) + 1)
        groups = [starts[k]:(starts[k+1]-1) for k in 1:length(starts)-1]
        order  = ["layer$k" for k in eachindex(groups)]
    end
    filter!(g -> length(g) ≥ 2, groups)
    isempty(groups) && throw(ArgumentError("$path: no layer has two or more samples"))

    # ── fit each layer ───────────────────────────────────────────────────────
    cρ = Vector{Float64}[]; κc = Function[]; μc = Function[]
    xb = Float64[0.0]; fluid = Bool[]; ns = Int[]; poor = String[]
    for (k, g) in enumerate(groups)
        p = sortperm(x[g])
        xs = x[g][p]; ρs = ρv[g][p]; Ks = Kv[g][p]; μs = μv[g][p]
        isfl = maximum(abs, μs) < μ_tol

        cd, ed = _lsq_fit(xs, ρs, poly_degree; rtol)
        ed < rtol || push!(poor, "$(order[k]) density (rel. RMS $(round(ed, sigdigits=2)))")
        push!(cρ, cd)

        # moduli fit on t = (x-xc)/Δ ∈ [-1,1], so a thin layer stays well conditioned
        xc = (first(xs) + last(xs))/2; Δ = max((last(xs) - first(xs))/2, eps())
        ts = (xs .- xc) ./ Δ
        for (q, ys, dest) in (("K", Ks, κc), ("mu", μs, isfl ? nothing : μc))
            dest === nothing && continue
            if all(isinf, ys)                       # incompressible layer
                push!(dest, _ -> Inf)
                continue
            end
            any(isinf, ys) && throw(ArgumentError(
                "$path: layer \"$(order[k])\" mixes finite and infinite $q; a layer is " *
                "either incompressible throughout or not at all"))
            c, e = _lsq_fit(ts, ys, poly_degree; rtol)
            e < rtol || push!(poor, "$(order[k]) $q (rel. RMS $(round(e, sigdigits=2)))")
            push!(dest, let xc=xc, Δ=Δ, c=c; y -> evalpoly((y - xc)/Δ, c) end)
        end
        isfl && push!(μc, _ -> 0.0)

        push!(fluid, isfl); push!(xb, last(xs))
        push!(ns, clamp(round(Int, 250*(last(xs) - first(xs))) + 20, 20, 160))
    end
    isempty(poor) || @warn "load_planet_csv: some layers were not resolved to rtol=$rtol; \
                            raise poly_degree or check the table" file=path layers=poor

    nL = length(groups); doms = [xb[i]..xb[i+1] for i in 1:nL]
    ρ̄ = 3*sum(poly_r2_integral(cρ[i], xb[i], xb[i+1]) for i in 1:nL)
    dens = [c ./ ρ̄ for c in cρ]; p_u = (4π*G_SI/3)*ρ̄^2*R^2   # same scheme as build_model
    κf = Function[let f = κc[i]; y -> (v = f(y); isinf(v) ? Inf : v/p_u) end for i in 1:nL]
    μf = Function[let f = μc[i]; y -> f(y)/p_u end for i in 1:nL]
    aw = [!fluid[i] for i in 1:nL]                 # fluid → non-AW (solid: ignored)
    Ω = Ω_SI !== nothing ? Float64(Ω_SI) :
        haskey(meta, "omega_si") ? something(tryparse(Float64, meta["omega_si"]), 0.0) : 0.0
    build_model(name, doms, dens, κf, μf, ns; R_SI = R, ρ̄_SI = ρ̄, aw = aw, Ω_SI = Ω)
end

"""
    planet_coeffs(model; degmax=12, rtol=1e-9) -> Vector{NamedTuple}

Recover each layer's `ρ, K, μ` as monomial coefficients in `x = r/R` (SI: kg/m³,
Pa), one entry `(layer, q, a, b, fluid, err, c)` each.

Fits at Chebyshev nodes with the lowest degree reaching relative RMS < `rtol`. A
genuinely polynomial profile terminates at its true degree with `err ~ 1e-16`; a
rational one (the Adams–Williamson outer-core `κ`) stops before the monomial
basis degrades and reports its honest `err`.
"""
function planet_coeffs(model::PlanetModel; degmax::Int = 12, rtol = 1e-9)
    R = model.R_SI; ρ̄ = model.ρ̄_SI; p_u = model.p_unit
    fits = NamedTuple[]
    for (i, l) in enumerate(model.layers)
        a, b = leftendpoint(l.domain), rightendpoint(l.domain)
        xs = [(a+b)/2 + (b-a)/2*cospi(k/240) for k in 0:240]
        for (q, f, u) in (("rho", l.ρ₀, ρ̄), ("K", l.κ, p_u), ("mu", l.μ, p_u))
            ys = [f(x)*u for x in xs]
            scale = maximum(abs, ys) + eps()
            rms(c) = sqrt(sum(abs2, [evalpoly(x, c) for x in xs] .- ys)/length(ys))/scale
            best_c, best_err = [ys[1]], rms([ys[1]])
            for d in 0:degmax
                c = ([x^j for x in xs, j in 0:d]) \ ys
                err = rms(c)
                if err < rtol                                     # exact polynomial
                    c[abs.(c) .< 1e-8*scale] .= 0.0               # snap roundoff junk
                    best_c, best_err = c, err
                    break
                end
                # skip ill-conditioned intermediate fits, but keep escalating —
                # an exact higher degree may still be reached
                maximum(abs, c) > 100*scale && continue
                err < best_err && ((best_c, best_err) = (c, err))
            end
            while length(best_c) > 1 && best_c[end] == 0.0; pop!(best_c); end
            push!(fits, (layer=i, q=q, a=a, b=b, fluid=is_fluid(l), err=best_err, c=best_c))
        end
    end
    return fits
end

"""
    write_planet_coeffs(model, path; degmax=12, rtol=1e-9) -> path

Write `planet_coeffs(model)` in the layout of PREM Table I: one block per layer
with its radius range, each quantity an explicit polynomial in `x = r/R`. SI
units. Exact polynomials print bare; fitted ones carry their relative RMS error.
"""
function write_planet_coeffs(model::PlanetModel, path; degmax::Int = 12, rtol = 1e-9)
    fits = planet_coeffs(model; degmax, rtol)
    R = model.R_SI
    poly(c) = isempty(c) || all(==(0.0), c) ? "0" :
        join(((k == 1 ? @sprintf("%.10g", ck) :
               ck < 0 ? @sprintf("- %.10g", -ck) : @sprintf("+ %.10g", ck)) *
              (k == 1 ? "" : k == 2 ? " x" : " x^$(k-1)")
              for (k, ck) in enumerate(c) if ck != 0.0), "  ")
    open(path, "w") do io
        @printf(io, "%s — polynomial profiles,  x = r/R,  R = %.1f km\n", model.name, R/1e3)
        println(io, "rho in kg/m^3, K and mu in Pa. Exact unless marked (fitted).")
        for i in 1:length(model.layers)
            lf = [f for f in fits if f.layer == i]
            @printf(io, "\nLayer %-2d  %8.1f – %8.1f km   %s\n", i,
                    lf[1].a*R/1e3, lf[1].b*R/1e3, lf[1].fluid ? "fluid" : "solid")
            for f in lf
                tag = f.err < rtol ? "" : @sprintf("   (fitted, rel err %.0e)", f.err)
                @printf(io, "  %-3s = %s%s\n", f.q, poly(f.c), tag)
            end
        end
    end
    return path
end
