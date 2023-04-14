module RandomPhaseApproximation

using Distributed: @distributed
using LinearAlgebra: I, Hermitian, NoPivot, cholesky, diagm, dot, eigen, norm
using QuantumLattices: AbstractLattice, Action, Algorithm, AnalyticalExpression, Assignment, BrillouinZone, CompositeIndex, Frontend, Hilbert, MatrixRepresentation, Neighbors, Operator, OperatorGenerator, Operators, OperatorUnitToTuple, RepresentationGenerator, ReciprocalSpace, ReciprocalZone, Table, Term
using QuantumLattices: plain, bonds, decimaltostr, dimension, expand, iscreation, kind, rank
using RecipesBase: RecipesBase, @recipe, @series
using Serialization: serialize
using SharedArrays: SharedArray
using TightBindingApproximation: AbstractTBA, Fermionic, TBA, TBAKind

import QuantumLattices: Parameters, add!, initialize, matrix, run!, update!

export EigenRPA, ParticleHoleSusceptibility, PHVertexMatrix, RPA, chiq, chiq0, chiq0chiq, correlation, eigenrpa, vertex

"""
    isevenperm(p::Vector) -> Bool

Judge the parity of permutations.
"""
function isevenperm(p::Vector)
    @assert isperm(p) "isevenperm error: invalid permutations."
    n = length(p)
    used = falses(n)
    even = true
    for k = 1:n
        if used[k]; continue; end
        # Each even cycle flips even (an odd number of times)
        used[k] = true
        j = p[k]
        while !used[j]
            used[j] = true
            j = p[j]
            even = !even
        end
    end
    return even
end

"""
    issamesite(op₁::CompositeIndex, op₂::CompositeIndex) -> Bool

Judge whether two composite indices are on site same site.
"""
@inline issamesite(op₁::CompositeIndex, op₂::CompositeIndex) = (op₁.index.site==op₂.index.site && op₁.icoordinate==op₂.icoordinate)

"""
    fermifunc(e::Real, temperature::Real=1e-12, μ::Real=0.0) -> Float64
 
Fermi distribution function. Boltzmann constant ``k_B=1``.
"""
function fermifunc(e::Real, temperature::Real=1e-12, μ::Real=0.0)
    if temperature > 1e-10
        f = (e-μ) / temperature
        if f > 20
            f = 0.0
        elseif f < -20
            f = 1.0
        else
            f = 1.0 / (1+exp(f))
        end
    else
        f = e - μ
        if f > 0.0
            f = 0.0
        else
            f = 1.0
        end
    end
    return f
end

"""
    PHVertexMatrix{D<:Number, Vq, Vk, T} <: MatrixRepresentation

Matrix representation of the particle-hole channel of two-body interaction terms:
```math
\\frac{1}{N}\\sum_{k₁k₂q, \\, \\alpha\\beta m n}[V^{ph}_{\\alpha\\beta, \\, mn}(q)-V^{ph}_{\\alpha m, \\, \\beta n}(k₂-k₁)]c^\\dagger_{k₁-q, \\, \\alpha}c_{k₁, \\, \\beta}c^\\dagger_{k₂, \\, n}c_{k₂-q, \\, m}
```
When the k₁ and k₂ are nothing, the exchange terms are omitted. Here, the Fourier transformation reads:
```math
c^†_i = \\frac{1}{\\sqrt{N}} ∑_k c^†_k \\exp(-i k rᵢ)
```
"""
struct PHVertexMatrix{D<:Number, Vq, Vk, T} <: MatrixRepresentation
    q::Vq
    k₁::Vk
    k₂::Vk
    table::T
    gauge::Symbol
    function PHVertexMatrix{D}(q, k₁, k₂, table, gauge::Symbol=:icoordinate) where {D<:Number}
        @assert gauge∈(:rcoordinate, :icoordinate) "PHVertexMatrix error: gauge must be `:rcoordinate` or `:icoordinate`."
        return new{D, typeof(q), typeof(k₁), typeof(table)}(q, k₁, k₂, table, gauge)
    end
end
@inline Base.valtype(mr::PHVertexMatrix) = valtype(typeof(mr))
@inline Base.valtype(::Type{<:PHVertexMatrix{D}}) where {D<:Number} = Matrix{promote_type(D, Complex{Int})}
@inline Base.valtype(R::Type{<:PHVertexMatrix}, ::Type{<:Union{Operator, Operators}}) = valtype(R)
@inline Base.zero(mr::PHVertexMatrix) = zeros(eltype(valtype(mr)), length(mr.table)^2, length(mr.table)^2)
@inline Base.zero(mr::PHVertexMatrix, ::Union{Operator, Operators}) = zero(mr)
@inline (mr::PHVertexMatrix)(m::Operator; kwargs...) = add!(zero(mr, m), mr, m; kwargs...)

"""
    PHVertexMatrix{D}(q, k₁, k₂, table, gauge::Symbol=:icoordinate) where {D<:Number}
    PHVertexMatrix{D}(table, gauge::Symbol=:icoordinate)  where {D<:Number}
    PHVertexMatrix{D}(q, table, gauge::Symbol=:icoordinate) where {D<:Number}

Get the matrix representation of particle-hole channel.
"""
@inline PHVertexMatrix{D}(table, gauge::Symbol=:icoordinate) where {D<:Number} = PHVertexMatrix{D}(nothing, nothing, nothing, table, gauge)
@inline PHVertexMatrix{D}(q, table, gauge::Symbol=:icoordinate) where {D<:Number} = PHVertexMatrix{D}(q, nothing, nothing, table, gauge)

"""
    add!(dest::AbstractMatrix, mr::PHVertexMatrix, m::Operator; kwargs...)

Get the matrix representation of an operator and add it to destination.
"""
function add!(dest::AbstractMatrix, mr::PHVertexMatrix, m::Operator; kwargs...)
    @assert rank(m) == 4 "add! error: rank(operator) $(rank(m)) != 4"
    n = length(mr.table)
    destr = zeros(eltype(dest), n, n, n, n)
    cr = Int[]
    an = Int[]
    for i = 1:rank(m)
        if iscreation(m[i])
            push!(cr, i)
        else
            push!(an, i)
        end
    end
    @assert length(cr)==length(an)==2 "add! error: not particle-hole pair interactions."
    p = [cr[1], an[1], cr[2], an[2]]
    sign = isevenperm(p) ? 1 : -1
    seq₁, seq₂, seq₃, seq₄ = mr.table[m[p[1]].index], mr.table[m[p[2]].index], mr.table[m[p[4]].index], mr.table[m[p[3]].index]
    if issamesite(m[p[1]], m[p[2]]) && issamesite(m[p[3]], m[p[4]]) && issamesite(m[p[2]], m[p[3]])
        destr[seq₁, seq₂, seq₃, seq₄] += m.value*sign
        destr[seq₄, seq₃, seq₂, seq₁] += m.value*sign
        destr[seq₁, seq₃, seq₂, seq₄] += -m.value*sign
        destr[seq₄, seq₂, seq₃, seq₁] += -m.value*sign
    elseif issamesite(m[p[1]], m[p[2]]) && issamesite(m[p[3]], m[p[4]])
        coordinate = mr.gauge==:rcoordinate ? m[p[3]].rcoordinate-m[p[1]].rcoordinate : m[p[3]].icoordinate-m[p[1]].icoordinate
        phaseq = isnothing(mr.q) ? one(eltype(dest)) : convert(eltype(dest), exp(-1im*dot(mr.q, coordinate)))
        destr[seq₁, seq₂, seq₃, seq₄] += m.value*sign*phaseq
        destr[seq₄, seq₃, seq₂, seq₁] += m.value*sign*conj(phaseq)
        if !(isnothing(mr.k₁)) && !(isnothing(mr.k₂)) # no test
            phasek = convert(eltype(dest), exp(-1im*dot(mr.k₂-mr.k₁, coordinate)))
            destr[seq₁, seq₃, seq₂, seq₄] += -m.value*sign*phasek
            destr[seq₄, seq₂, seq₃, seq₁] += -m.value*sign*conj(phasek)
        end
    elseif issamesite(m[p[1]], m[p[4]]) && issamesite(m[p[3]], m[p[2]])
        coordinate = mr.gauge==:rcoordinate ? m[p[3]].rcoordinate-m[p[1]].rcoordinate : m[p[3]].icoordinate-m[p[1]].icoordinate
        phaseq = isnothing(mr.q) ? one(eltype(dest)) : convert(eltype(dest), exp(-1im*dot(mr.q, coordinate)))
        destr[seq₁, seq₃, seq₂, seq₄] += -m.value*sign*phaseq
        destr[seq₄, seq₂, seq₃, seq₁] += -m.value*sign*conj(phaseq)
        if !(isnothing(mr.k₁)) && !(isnothing(mr.k₂)) # no test
            phasek = convert(eltype(dest), exp(-1im*dot(mr.k₂-mr.k₁, coordinate)))
            destr[seq₁, seq₂, seq₃, seq₄] += m.value*sign*phasek
            destr[seq₄, seq₃, seq₂, seq₁] += m.value*sign*conj(phasek)
        end
    else
        error("add! error: the two-body interaction is error.")
    end
    dest[:, :] += reshape(destr, n^2, n^2)
    return dest
end

"""
    RPA{L<:AbstractTBA, U<:RepresentationGenerator} <: Frontend

Random phase approximation in a fermionic system.
"""
struct RPA{L<:AbstractTBA, U<:RepresentationGenerator} <: Frontend
    tba::L
    interactions::U
end
@inline Parameters(rpa::RPA) = merge(Parameters(rpa.tba), Parameters(rpa.interactions))
@inline function update!(rpa::RPA; k=nothing, kwargs...)
    if length(kwargs)>0
        update!(rpa.tba.H; kwargs...)
        update!(rpa.interactions; kwargs...)
    end
    return rpa
end

"""
    RPA(tba::AbstractTBA, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing)
    RPA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing)
    RPA(tba::AbstractTBA{K, <:AnalyticalExpression}, hilbert::Hilbert, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing) where {K<:TBAKind}

Construct an `RPA` type.
"""
function RPA(tba::AbstractTBA, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing)
    @assert tba.H.operators.boundary==plain "RPA error: unsupported boundary condition."
    table = Table(tba.H.hilbert, OperatorUnitToTuple(:site, :orbital, :spin))
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, interactions))
    int = OperatorGenerator(interactions, bonds(tba.lattice, neighbors), tba.H.hilbert; table=table)
    return RPA(tba, int)
end
@inline function RPA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing)
    return RPA(TBA(lattice, hilbert, terms; neighbors=neighbors), interactions)
end
function RPA(tba::AbstractTBA{K, <:AnalyticalExpression}, hilbert::Hilbert, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing) where {K<:TBAKind}
    K<:Fermionic{:BdG} && @warn "the table of tba should be (:nambu, *, *, *) where * denotes other degrees of freedom."
    table = Table(hilbert, OperatorUnitToTuple(:site, :orbital, :spin))
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, interactions))
    int = OperatorGenerator(interactions, bonds(tba.lattice, neighbors), hilbert; table=table)
    return RPA(tba, int)
end

"""
    matrix(rpa::RPA, field::Symbol=:int; k=nothing, gauge=:icoordinate, kwargs...) -> Matrix

Get matrix of particle-hole channel of interaction.
"""
@inline function matrix(rpa::RPA, field::Symbol=:int; k=nothing, gauge=:icoordinate, kwargs...)
    field==:tba && return matrix(rpa.tba; k=k, gauge=gauge, kwargs...)
    field==:int && return PHVertexMatrix{valtype(eltype(rpa.interactions))}(k, rpa.interactions.table, gauge)(expand(rpa.interactions))
    error("matrix error: wrong field.")
end

"""
    vertex(rpa::RPA, reciprocalspace::AbstractVector{<:AbstractVector}, gauge=:icoordinate) -> Array{<:Number, 3}

Return particle-hole vertex induced by the direct channel of interaction (except the Hubbard interaction which include both direct and exchange channels).
"""
@inline function vertex(rpa::RPA, reciprocalspace::AbstractVector{<:AbstractVector}, gauge=:icoordinate)
    n, nq = length(rpa.interactions.table), length(reciprocalspace)
    result = zeros(promote_type(Complex{Int}, valtype(eltype(rpa.interactions))), n^2, n^2, nq)
    for (i, q) in enumerate(reciprocalspace)
        result[:, :, i] = matrix(rpa, :int; k=q, gauge=gauge)
    end
    return result
end

"""
    ParticleHoleSusceptibility{P<:ReciprocalSpace, B<:BrillouinZone, E<:AbstractVector, S<:Operators} <: Action

Calculate the particle-hole susceptibility within random phase approximation.

Attribute `options` contains `(η=0.01, gauge=:icoordinate, temperature=1e-12, μ=0.0, findk=false)`.
"""
struct ParticleHoleSusceptibility{P<:ReciprocalSpace, B<:BrillouinZone, E<:AbstractVector, S<:Operators} <: Action
    reciprocalspace::P
    brillouinzone::B
    energies::E
    operators::Tuple{Vector{S}, Vector{S}}
    options::Dict{Symbol, Any}
    function ParticleHoleSusceptibility(reciprocalspace::ReciprocalSpace, brillouinzone::BrillouinZone, energies::AbstractVector, operators::Tuple{Vector{S}, Vector{S}}, options::Dict{Symbol, Any}) where {S<:Operators}
        @assert names(reciprocalspace)==(:k,) "ParticleHoleSusceptibility error: the name of the momenta must be :k."
        @assert length(operators[1])==length(operators[2]) "ParticleHoleSusceptibility error: the number of left operators must be equal to that of right operators."
        new{typeof(reciprocalspace), typeof(brillouinzone), typeof(energies), S}(reciprocalspace, brillouinzone, energies, operators, options)
    end
end

"""
    ParticleHoleSusceptibility(reciprocalspace::ReciprocalSpace, brillouinzone::BrillouinZone, energies::Vector, operators::Tuple{Vector{<:Operators}, Vector{<:Operators}}; options...)

Construct a `ParticleHoleSusceptibility` type.
"""
@inline function ParticleHoleSusceptibility(reciprocalspace::ReciprocalSpace, brillouinzone::BrillouinZone, energies::AbstractVector, operators::Tuple{Vector{<:Operators}, Vector{<:Operators}}; options...)
    return ParticleHoleSusceptibility(reciprocalspace, brillouinzone, energies, operators, convert(Dict{Symbol, Any}, options))
end

@inline function initialize(phs::ParticleHoleSusceptibility, rpa::RPA)
    x = collect(Float64, 0:(length(phs.reciprocalspace)-1))
    y = collect(Float64, phs.energies)
    z = zeros(ComplexF64, length(y), length(x))
    z₀ = zeros(ComplexF64, length(y), length(x))
    return (x, y, z, z₀)
end
function run!(rpa::Algorithm{<:RPA}, phs::Assignment{<:ParticleHoleSusceptibility})
    gauge = get(phs.action.options, :gauge, :icoordinate)
    η = get(phs.action.options, :η, 0.01)
    temperature = get(phs.action.options, :temperature, 1e-12)
    μ = get(phs.action.options, :μ, 0.0)
    scflag = kind(rpa.frontend.tba)==Fermionic(:BdG) ? true : false
    findk = get(phs.action.options, :findk, false)
    vph = vertex(rpa.frontend, phs.action.reciprocalspace, gauge)
    if findk
        ndim, nk = dimension(rpa.frontend.tba), length(phs.action.brillouinzone)
        eigvecs = zeros(ComplexF64, ndim, ndim, nk)
        eigvals = zeros(Float64, ndim, nk)
        for (i, k) in enumerate(phs.action.brillouinzone)
            eigensystem = eigen(matrix(rpa.frontend.tba; k=k, gauge=:icoordinate))
            eigvals[:, i] = eigensystem.values
            eigvecs[:, :, i] = eigensystem.vectors
        end
        χ₀, χ = chiq0chiq(eigvecs, eigvals, phs.action.brillouinzone, phs.action.reciprocalspace, vph, phs.action.energies; η=η, temperature=temperature, μ=μ, scflag=scflag)
    else
        ecut = get(phs.action.options, :cut_off, Inf)
        if ecut < Inf
            @warn "This mode (cut off energy) is the experimental method."
            brillouinzone = _kpoint_cutoff(phs.action.brillouinzone, ecut, μ, rpa.frontend.tba)
            χ₀, χ = chiq0chiq(rpa.frontend.tba, brillouinzone, phs.action.reciprocalspace, vph, phs.action.energies; η=η, temperature=temperature, μ=μ, scflag=scflag, gauge=gauge, phs.action.options...)
        else
            χ₀, χ = chiq0chiq(rpa.frontend.tba, phs.action.brillouinzone, phs.action.reciprocalspace, vph, phs.action.energies; η=η, temperature=temperature, μ=μ, scflag=scflag, gauge=gauge, phs.action.options...)
        end
    end
    get(phs.action.options, :save, false) && (serialize(join([get(phs.action.options, :filename, "chi"), "0"]), χ₀))
    gauge = findk ? :rcoordinate : gauge==:rcoordinate ? :icoordinate : :rcoordinate
    phs.data[3][:, :] = correlation(χ, phs.action.reciprocalspace, phs.action.operators, rpa.frontend.interactions.table; gauge=gauge)
    phs.data[4][:, :] = correlation(χ₀, phs.action.reciprocalspace, phs.action.operators, rpa.frontend.interactions.table; gauge=gauge)
end
function _kpoint_cutoff(brillouinzone::BrillouinZone, ecut::Float64, μ::Float64, tba::AbstractTBA)
    result = eltype(brillouinzone)[]
    for k in brillouinzone
        eigensystem = eigen(matrix(tba; k=k, gauge=:icoordinate))
        minimum(abs, eigensystem.values .- μ)>ecut || push!(result, k)
    end
    return result
end

"""
    correlation(χ::Array{<:Number, 4}, reciprocalspace::ReciprocalSpace, operators::Tuple{Vector{<:Operators}, Vector{<:Operators}}, table; gauge=:rcoordinate) -> Matrix

Return physical particle-hole susceptibility.
"""
function correlation(χ::Array{<:Number, 4}, reciprocalspace::ReciprocalSpace, operators::Tuple{Vector{<:Operators}, Vector{<:Operators}}, table; gauge=:rcoordinate)
    @assert length(operators[1])==length(operators[2]) "correlation error: the number of left operators must be equal to that of right operators."
    n = length(table)
    m₁ = zeros(ComplexF64, n, n)
    m₂ = zeros(ComplexF64, n, n)
    nw, nq = size(χ, 3), size(χ, 4)
    χm = reshape(χ, n, n, n, n, nw, nq)
    nop = length(operators[1])
    result = zeros(eltype(χ), nw, nq)
    for (i, momentum) in enumerate(reciprocalspace)
        for l in 1:nop
            ops₁ = operators[1][l]
            ops₂ = operators[2][l]
            matrix!(m₁, ops₁, table, momentum; gauge=gauge)
            matrix!(m₂, ops₂, table, -momentum; gauge=gauge)
            for a in 1:n
                for b in 1:n
                    for c in 1:n
                        for d in 1:n
                            result[:, i] += m₁[b, a] * m₂[c, d] * χm[a, b, c, d, :, i]
                        end
                    end
                end
            end
        end
    end
    return result
end
function matrix!(m::Matrix{<:Number}, operators::Operators, table::Table, k; gauge=:rcoordinate)
    m[:, :] .= zero(eltype(m))
    for op in operators
        phase = gauge==:icoordinate ? one(eltype(m)) : convert(eltype(m), exp(1im*dot(k, op[1].rcoordinate-op[1].icoordinate)))
        seq₁ = table[op[1].index']
        seq₂ = table[op[2].index]
        m[seq₁, seq₂] += op.value * phase
    end
    return m
end

# Return ``chi^0(k, q)_{ij, mn}=chi^0(k, q)_{-+,+-}``, where ``chi^0(k, q)_{12, 34} ≡ <c^\\dagger_{k, 2}c_{k-q, 1}c^\\dagger_{k-q, 3}c_{k, 4}>``.
@inline function _chikq0(tba::AbstractTBA, k::AbstractVector, q::AbstractVector, omega::Float64; scflag::Bool, η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, kwargs...)
    Ek, uk = eigen(Hermitian(matrix(tba; k=k, kwargs...)))
    Ekq, ukq = eigen(Hermitian(matrix(tba; k=k-q, kwargs...)))
    return _chikq0(Ek, Ekq, uk, ukq, omega; scflag=scflag, η=η, temperature=temperature, μ=μ)
end
function _chikq0(Ek::Vector{Float64}, Ekq::Vector{Float64}, uk::Matrix{<:Number}, ukq::Matrix{<:Number}, omega::Float64; scflag::Bool, η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0)
    n = length(Ek)
    if scflag
        @assert iseven(n) "_chikq0 error: odd dimension when pairing terms exist."
        temp = diagm([(fermifunc(Ek[i], temperature, μ)-fermifunc(Ekq[j], temperature, μ))/(omega+η*im+Ek[i]-Ekq[j]) for i=1:n for j=1:n])
        m = Int(n/2)
        u = kron(conj(uk[1:m, :]), ukq[1:m, :])
        v = reshape(permutedims(reshape(kron(uk[m+1:n, :], conj(ukq[m+1:n, :])), (m, m, n*n)), [3, 2, 1]), (n*n, m*m))
        return -u*temp*adjoint(u) + u*temp*v
    else
        temp = diagm([-(fermifunc(Ek[i], temperature, μ)-fermifunc(Ekq[j], temperature, μ))/(omega+η*im+Ek[i]-Ekq[j]) for i=1:n for j=1:n])
        u = kron(conj(uk), ukq)
        return u*temp*adjoint(u)
    end
end

"""
    chiq0chiq(
        tba::AbstractTBA, brillouinzone::AbstractVector{<:AbstractVector}, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 3}, energies::AbstractVector;
        η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false, kwargs...
    ) -> Tuple{Array{ComplexF64, 4}, Array{ComplexF64, 4}}

Get the particle-hole susceptibilities χ⁰(ω, q) and χ(ω, q). The spectral function is satisfied by ``A(ω, q) = \\text{Im}[χ(ω+i0⁺, q)]``.

# Arguments
- `vph`: the bare particle-hole vertex (vph[ndim, ndim, nq])
- `energies`: the energy points
- `η`: the magnitude of broaden
- `temperature`: the temperature
- `μ`: the chemical potential
- `scflag`: false (default, no superconductivity), or true (BdG model)
- `kwargs`: the keyword arguments transferred to the `matrix(tba; kwargs...)` function
"""
function chiq0chiq(
    tba::AbstractTBA, brillouinzone::AbstractVector{<:AbstractVector}, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 3}, energies::AbstractVector;
    η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false, kwargs...
)
    nk = length(brillouinzone)
    nq = length(reciprocalspace)
    nw = length(energies)
    ndim = size(vph, 1)
    chi0 = SharedArray{ComplexF64}(ndim, ndim, nw, nq)
    chi = SharedArray{ComplexF64}(ndim, ndim, nw, nq)
    idmat = Matrix{Float64}(I, ndim, ndim)
    @sync @distributed for j = 1:nq*nw #iq=1:nq
        iq, iw = fldmod1(j, nw)
        q = reciprocalspace[iq]
        ω = energies[iw]
        chi0[:, :, iw, iq] = @distributed (+) for i = 1:nk
            1/nk*_chikq0(tba, brillouinzone[i], q, ω; scflag=scflag, η=η, temperature=temperature, μ=μ, kwargs...)
        end
        chi[:, :, iw, iq] = chi0[:, :, iw, iq] * inv(idmat+vph[:, :, iq]*chi0[:, :, iw, iq])
    end
    return Array(chi0), Array(chi)
end

"""
    chiq0chiq(
        eigvecs::Array{ComplexF64, 3}, eigvals::Array{Float64, 2}, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 3}, energies::Vector{Float64};
        η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false
    ) -> Tuple{Array{ComplexF64, 4}, Array{ComplexF64, 4}}

Get the particle-hole susceptibilities χ⁰(ω, q) and χ(ω, q). The spectral function is satisfied by ``A(ω, q) = \\text{Im}[χ(ω+i0⁺, q)]``.
"""
function chiq0chiq(
    eigvecs::Array{ComplexF64,3}, eigvals::Array{Float64,2}, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 3}, energies::AbstractVector;
    η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false
)
    nk = length(brillouinzone)
    nq = length(reciprocalspace)
    nw = length(energies)
    ndim = size(vph, 1)
    chi0 = SharedArray{ComplexF64}(ndim, ndim, nw, nq)
    chi = SharedArray{ComplexF64}(ndim, ndim, nw, nq)
    idmat = Matrix{Float64}(I, ndim, ndim)
    @sync @distributed for j = 1:nq*nw
        iq, iw = fldmod1(j, nw)
        q = reciprocalspace[iq]
        ω = energies[iw]
        chi0[:, :, iw, iq] = @distributed (+) for i = 1:nk
            ikq = Int(keytype(brillouinzone)(brillouinzone[i]-q, brillouinzone.reciprocals))
            1/nk*_chikq0(eigvals[:, i], eigvals[:, ikq], eigvecs[:, :, i], eigvecs[:, :, ikq], ω; scflag=scflag, η=η, temperature=temperature, μ=μ)
        end
        chi[:, :, iw, iq] = chi0[:, :, iw, iq] * inv(idmat+vph[:, :, iq]*chi0[:, :, iw, iq])
    end
    return Array(chi0), Array(chi)
end

"""
    chiq0(tba::AbstractTBA, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, energies::Vector{Float64}; η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false, kwargs...) -> Array{Float64, 4}

Get the particle-hole susceptibilities χ⁰(ω, q).

# Arguments
- `energies`: the energy points
- `η`: the magnitude of broaden
- `temperature`: the temperature
- `μ`: the chemical potential
- `scflag`: false(default) for particle-hole channel, true for Nambu space
- `kwargs`: the keyword arguments transferred to the `matrix(tba; kwargs...)` function
"""  
function chiq0(tba::AbstractTBA, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, energies::AbstractVector; η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false, kwargs...)
    nk = length(brillouinzone)
    nq = length(reciprocalspace)
    nw = length(energies)
    ndim = dimension(tba)
    chi0 = zeros(ComplexF64, ndim^2, ndim^2, nw, nq)
    for iq = 1:nq
        for iw = 1:nw
            for i = 1:nk
                chi0[:, :, iw, iq] += 1/nk*_chikq0(tba, brillouinzone[i], reciprocalspace[iq], energies[iw]; scflag=scflag, η=η, temperature=temperature, μ=μ, kwargs...)
            end
        end
    end
    return chi0
end

"""
    chiq(vph::Array{<:Number, 3}, chi0::Array{ComplexF64, 4}) -> Array{ComplexF64, 4}

Get the susceptibility ``χ_{αβ}(ω, q)``.
"""
function chiq(vph::Array{<:Number, 3}, chi0::Array{ComplexF64,4})
    nq = size(chi0, 4)
    nw = size(chi0, 3)
    ndim = size(vph, 1)
    chi = zeros(ComplexF64, ndim, ndim, nw, nq)
    idmat = Matrix{Float64}(I, ndim, ndim)
    for iq = 1:nq
        for iw = 1:nw
            chi[:, :, iw, iq] = chi0[:, :, iw, iq] * inv(idmat+vph[:, :, iq]*chi0[:, :, iw, iq])
        end
    end
    return chi
end

"""
    @recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}}, mode::Symbol=:χ)

Define the recipe for the visualization of particle-hole susceptibilities.
"""
@recipe function plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}}, mode::Symbol=:χ)
    title --> nameof(pack[1], pack[2])
    titlefontsize --> 10
    legend --> true
    @assert mode in (:χ, :χ0) "plot error: mode must be one of (:χ, :χ0)."
    if mode == :χ
        χim0 = imag.(pack[2].data[3])/pi
        colorbar_title := "χ(q, ω)"
    elseif mode == :χ0
        χim0 = imag.(pack[2].data[4])/pi
        colorbar_title := "χ₀(q, ω)"
    end
    clims = extrema(χim0)
    xlabel := "q"
    ylabel := "ω"
    @series begin
        seriestype := :heatmap
        clims --> clims
        pack[2].data[1], pack[2].data[2], χim0
    end
end

"""
    @recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}}, ecut::Float64, dE::Float64=1e-3, mode::Symbol=:χ, reim::Symbol=:re)

"""
@recipe function plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}}, ecut::Float64, dE::Float64=1e-3, mode::Symbol=:χ, reim::Symbol=:re)
    title --> nameof(pack[1], pack[2])
    titlefontsize --> 10
    xlabel --> "q₁"
    ylabel --> "q₂"
    aspect_ratio := :equal
    colorbar_title := string(reim==:re ? "Re" : "Im", mode, "(q, ω=", decimaltostr(ecut), ")")
    @series begin
        seriestype := :heatmap
        data = spectralecut(pack[2], ecut, dE, mode)
        reim==:re && (data = (data[1], data[2], real.(data[3])))
        reim==:im && (data = (data[1], data[2], imag.(data[3])/pi))
        clims --> extrema(data[3])
        xlims --> (minimum(data[1]), maximum(data[1]))
        ylims --> (minimum(data[2]), maximum(data[2]))
        data[1], data[2], data[3]
    end
end
function spectralecut(ass::Assignment{<:ParticleHoleSusceptibility}, ecut::Float64, dE::Float64, mode::Symbol=:χ)
    @assert isa(ass.action.reciprocalspace, ReciprocalZone) "spectralecut error: please input an instance of ReciprocalZone."
    energies = ass.data[2]
    f(x) = abs(x-ecut) <= dE ? true : false
    idx = findall(f, energies)
    if mode == :χ
        intensity = ass.data[3]
    elseif mode == :χ0
        intensity = ass.data[4]
    end
    dims = Int[]
    seg = []
    reciprocals = []
    for (i, bound) in enumerate(ass.action.reciprocalspace.bounds)
        if bound.length > 1
            push!(dims, bound.length)
            push!(seg, bound)
            push!(reciprocals, ass.action.reciprocalspace.reciprocals[i])
        end
    end
    @assert length(dims)==2 "spectralecut error: the k points is not in a plane."
    y = collect(seg[2])*norm(reciprocals[2]) #collect(Float64, 0:(dims[2]-1))
    x = collect(seg[1])*norm(reciprocals[1]) #collect(Float64, 0:(dims[1]-1))
    z = reshape(sum(intensity[idx, :], dims=1), reverse(dims)...)
    return (x, y, z)
end

# eigen problem of RPA
"""
    EigenRPA{P<:ReciprocalSpace, B<:BrillouinZone} <: Action

Eigen problem for standard random phase approximation.
"""
struct EigenRPA{P<:ReciprocalSpace, B<:BrillouinZone} <: Action
    reciprocalspace::P
    brillouinzone::B
    eigvals_only::Bool
    options::Dict{Symbol, Any}
    function EigenRPA(reciprocalspace::ReciprocalSpace, brillouinzone::BrillouinZone, eigvals_only::Bool, options::Dict{Symbol, Any})
        @assert names(reciprocalspace) == (:k,) "ParticleHoleSusceptibility error: the name of the momenta must be :k."
        new{typeof(reciprocalspace), typeof(brillouinzone)}(reciprocalspace, brillouinzone, eigvals_only, options)
    end
end

"""
    EigenRPA(reciprocalspace::ReciprocalSpace, brillouinzone::BrillouinZone; eigvals_only::Bool=true, options...)

Construct a `EigenRPA` type. Attribute `options` contains `(gauge=:icoordinate, exchange=false, η=1e-8, temperature=1e-12, μ=0.0, bands=nothing)`.
"""
@inline function EigenRPA(reciprocalspace::ReciprocalSpace, brillouinzone::BrillouinZone; eigvals_only::Bool=true, options...)
    return EigenRPA(reciprocalspace, brillouinzone, eigvals_only, convert(Dict{Symbol, Any}, options))
end

@inline function initialize(erpa::EigenRPA, rpa::RPA)
    x = collect(Float64, 0:(length(erpa.reciprocalspace)-1))
    eigvals = Vector{ComplexF64}[]
    eigvecs = Matrix{ComplexF64}[]
    o2bs = Matrix{ComplexF64}[]
    return x, eigvals, eigvecs, o2bs
end
function run!(rpa::Algorithm{<:RPA}, erpa::Assignment{<:EigenRPA})
    int, tba = rpa.frontend.interactions, rpa.frontend.tba
    n = length(int.table)
    nq = length(erpa.action.reciprocalspace)
    nk = length(erpa.action.brillouinzone)
    vph = zeros(ComplexF64, n^2, n^2, nk, nk, nq)
    gauge = get(erpa.action.options, :gauge, :icoordinate)
    if get(erpa.action.options, :exchange, false)
        ops = expand(int)
        for (i, q) in enumerate(erpa.action.reciprocalspace)
            for (l, k₂) in enumerate(erpa.action.brillouinzone)
                for (j, k₁) in enumerate(erpa.action.brillouinzone)
                    vph[:, :, j, l, i] = PHVertexMatrix{valtype(eltype(int))}(q, k₁, k₂, int.table, gauge=gauge)(ops)
                end
            end  
        end
    else
        for (i, q) in enumerate(erpa.action.reciprocalspace)
            m = matrix(rpa.frontend, :int; k=q, gauge=gauge)
            for j in 1:nk
                for l in 1:nk
                    vph[:, :, j, l, i] = m
                end
            end
        end
    end
    η = get(erpa.action.options, :η, 1e-8)
    temperature = get(erpa.action.options, :temperature, 1e-12)
    μ = get(erpa.action.options, :μ, 0.0)
    bands = get(erpa.action.options, :bands, nothing)
    eigvals, eigvecs, o2bs = eigenrpa(tba, erpa.action.brillouinzone, erpa.action.reciprocalspace, vph; temperature=temperature, μ=μ, eigvals_only=erpa.action.eigvals_only, η=η, bands=bands, gauge=gauge, erpa.action.options...)
    append!(erpa.data[2], eigvals)
    append!(erpa.data[3], eigvecs)
    append!(erpa.data[4], o2bs)
end

"""
    eigenrpa(
        tba::AbstractTBA, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 5};
        temperature::Float64=1e-12, μ::Float64=0.0, eigvals_only::Bool=true, η::Float64=1e-6, bands::Union{UnitRange{Int}, StepRange{Int,Int}, Vector{Int}, Nothing}=nothing, kwargs...
    ) -> Tuple{Array{Array{ComplexF64, 1}, 1}, Array{Matrix{ComplexF64}, 1}, Array{Matrix{ComplexF64}, 1}}


Get the eigenvalues, eigenvectors, and unitary transformation (from orbital to band) of particle-hole susceptibilities ``χ_{αβ}(ω, k₁, k₂, q)``.

Now only the zero-temperature case is supported.

# Arguments
- `vph`: the particle-hole vertex, e.g. vph[ndim,ndim,nk,nk,nq] where sqrt(ndim) is the number of degrees of freedom in the unit cell, nk=length(brillouinzone), nq=length(reciprocalspace)
- `temperature`: the temperature
- `μ`: the chemical potential
- `eigvals_only`: only the eigenvalues needed, when it is false the cholesky method is used
- `η`: the small number to avoid the semi-positive Hamiltonian, i.e. Hamiltonian+diagm([η, η, ...])
- `bands`: the selected bands to calculate the χ₀
- `kwargs`: the keyword arguments transferred to the `matrix(tba; kwargs...)` function
"""
function eigenrpa(
    tba::AbstractTBA, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 5};
    temperature::Float64=1e-12, μ::Float64=0.0, eigvals_only::Bool=true, η::Float64=1e-6, bands::Union{UnitRange{Int}, StepRange{Int, Int}, Vector{Int}, Nothing}=nothing, kwargs...
)
    nk = length(brillouinzone)
    nq = length(reciprocalspace)
    eigvals = Array{ComplexF64,1}[]
    eigvecs = Array{ComplexF64,2}[]
    o2bs = Array{ComplexF64,2}[]
    for iq = 1:nq
        chi0inv = Float64[]
        um = Array{ComplexF64,2}[]
        list = Int[1]
        g = Float64[]
        for i = 1:nk
            temp0, tempu, g0 = _chikq03(tba, brillouinzone[i], reciprocalspace[iq], 0.0; temperature=temperature, μ=μ, bands=bands, kwargs...)
            push!(um, tempu)
            append!(chi0inv, temp0)
            append!(g, g0)
            push!(list, size(tempu, 2))
        end
        list = cumsum(list)
        wk = zeros(ComplexF64, list[end]-1, list[end]-1)
        for i = 1:nk
            for j = 1:nk
                wk[list[j]:list[j+1]-1, list[i]:list[i+1]-1] = transpose(conj(um[j]))*vph[:, :, j, i, iq]*um[i]/nk
            end
        end
        heff = Hermitian(-diagm(chi0inv)+wk, :U)
        if eigvals_only
            eigensystem = eigen(heff*diagm(g))
        else
            ck = cholesky(heff+diagm([η for i=1:size(heff, 1)]), NoPivot(); check=true)
            eigensystem = eigen(ck.U*diagm(g)*ck.L)
            push!(eigvecs, inv(ck.U)*eigensystem.vectors*diagm(sqrt.(abs.(eigensystem.values))))
        end
        push!(eigvals, eigensystem.values)
        push!(o2bs, hcat(um...))
    end
    return eigvals, eigvecs, o2bs
end
function _chikq03(tba::AbstractTBA, k::AbstractVector, q::AbstractVector, omega::Float64=0.0; temperature::Float64=1e-12, μ::Float64=0.0, bands::Union{UnitRange{Int}, StepRange{Int, Int}, Vector{Int}, Nothing}=nothing, kwargs...)
    @assert temperature < 1e-10 "_chikq03 error: now only support the zero temperature (<1e-10) $(temperature). "
    n = dimension(tba)
    isnothing(bands) && (bands = 1:n)
    Ek, uk = eigen(Hermitian(matrix(tba; k=k, kwargs...)))
    Ekq, ukq = eigen(Hermitian(matrix(tba; k=k-q, kwargs...)))
    ee1, lit1, one1 = Float64[], Int[], Float64[]
    ee2, lit2, one2 = Float64[], Int[], Float64[]
    for i in bands
        for j in bands
            ft = fermifunc(Ek[i], temperature, μ) - fermifunc(Ekq[j], temperature, μ)
            if abs(ft) > eps()
                if Ek[i] - Ekq[j] < 0.0
                    push!(lit1, (i-1)*n+j)
                    push!(ee1, (omega+Ek[i]-Ekq[j])/ft)
                    push!(one1, 1.0)
                else Ek[i] - Ekq[j] > 0.0
                    push!(lit2, (i-1)*n+j)
                    push!(ee2, (omega+Ek[i]-Ekq[j])/ft)
                    push!(one2, -1.0)
                end
            end
        end
    end
    u = kron(conj(uk), ukq)
    ee = vcat(ee1, ee2)
    uu = zeros(ComplexF64, n*n, length(ee))
    uu[:, 1:length(ee1)] = u[:, lit1]
    uu[:, length(ee1)+1:end] = u[:, lit2]
    return ee, uu, vcat(one1, one2)
end

"""
    chiq(eigvals::Vector{<:Vector{<:Number}}, eigvecs::Vector{<:Matrix{<:Number}}, o2bs::Vector{<:Matrix{<:Number}}, energies::AbstractVector; η::Float64=1e-2, imag_only::Bool=false) -> Array{ComplexF64, 4}

Get the ``\\chi_{ij, nm}(\\omega, q)``. When `imag_only` is true, only the imaginary part is calculated.

Here, the eigenvalues, eigenvectors, and the orbital-to-band unitary matrices should be obtained by the method `eigenrpa`.
"""
function chiq(eigvals::Vector{<:Vector{<:Number}}, eigvecs::Vector{<:Matrix{<:Number}}, o2bs::Vector{<:Matrix{<:Number}}, energies::AbstractVector; η::Float64=1e-2, imag_only::Bool=false)
    nq = length(eigvals)
    nw = length(energies)
    ndim = size(o2bs[1], 1)
    chi = zeros(ComplexF64, ndim, ndim, nw, nq)
    if imag_only
        for iq = 1:nq
            for iw = 1:nw
                for i = 1:length(eigvals[iq])
                    temp = o2bs[iq] * eigvecs[iq][:, i]
                    chi[:, :, iw, iq] += temp * η / ((energies[iw]-eigvals[iq][i])^2+η^2) * adjoint(temp)
                end
            end
        end
    else
        n = size(o2bs[1], 2)
        for iq = 1:nq
            for iw = 1:nw
                tm = eigvecs[iq]
                tmu = o2bs[iq]*tm
                chi[:, :, iw, iq] = tmu * diagm([1/(-energies[iw]-im*η+eigvals[iq][i]) for i=1:n]) * (diagm(sign.(eigvals[iq]))) * adjoint(tmu)
            end
        end
    end
    return chi
end

"""
    @recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:EigenRPA}})

Define the recipe for the visualization of particle-hole susceptibilities.
"""
@recipe function plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:EigenRPA}}, reim::Symbol=:real)
    title --> nameof(pack[1], pack[2])
    titlefontsize --> 10
    legend --> false
    n = length(pack[2].data[2][end])
    nq = length(pack[2].data[2])
    eigvals = zeros(ComplexF64, nq, n)
    for (i, v) in enumerate(pack[2].data[2])
        eigvals[i, :] = v
    end
    values = reim==:real ? real.(eigvals) : imag.(eigvals)
    xlabel := "q"
    ylabel := "ω"
    minorgrid --> true
    showaxis --> :yes
    @series begin
        seriestype := :path
        pack[2].data[1], values
    end
    @series begin
        seriestype := :scatter
        pack[2].data[1], values
    end
end

end # module
