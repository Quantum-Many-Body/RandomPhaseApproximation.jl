module RandomPhaseApproximation
using Distributed: @distributed 
using QuantumLattices: ID, MatrixRepresentation, Operator, Operators, RepresentationGenerator, rank, creation, annihilation, CompositeIndex, Frontend, Hilbert, Term, Neighbors, Boundary, plain
using QuantumLattices: OperatorUnitToTuple, Table, OperatorGenerator, bonds, expand, kind, dimension, ReciprocalPath, ReciprocalZone, Action, Algorithm, Assignment
using QuantumLattices:  kind, dimension, AbstractLattice
import QuantumLattices: add!, matrix, update!, Parameters, initialize, run!
using TightBindingApproximation: AbstractTBA, TBA, Fermionic
using LinearAlgebra: diagm, eigen, Hermitian, cholesky, I, dot, NoPivot
using SharedArrays: SharedArray
using DelimitedFiles: writedlm
using Serialization: serialize
using RecipesBase: RecipesBase, @recipe, @series

export RPA, EigenRPA, chiq, chikqm, chiq0, correlation, findk, projchi, projchiim, fermifunc
export PHVertexRepresentation, isevenperm, issamesite, ParticleHoleSusceptibility, selectpath

"""
    isevenperm(p::Vector) -> Bool

Judge the number of permutations.
"""
function isevenperm(p::Vector)
    @assert isperm(p)
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
    even
end
issamesite(op₁::CompositeIndex, op₂::CompositeIndex) = (op₁.index.site == op₂.index.site && op₁.icoordinate == op₂.icoordinate)

"""
    PHVertexRepresentation{H<:RepresentationGenerator, Vq, Vk, T} <: MatrixRepresentation

Matrix representation of the particle-hole channel of two-body interaction terms. When the k₁ and k₂ are nothing, the exchange terms is ommitted. 
``1/N\\sum_{kk'q\\alpha\beta m n}[V^{ph}_{\\alpha\\beta,mn}(q)-V^{ph}_{\\alpha m,\\beta n}(k'-k)]c^\\dagger_{k-q\\alpha}c_{k\\beta}c^\\dagger_{k' n}c_{k'-q m}``
c^†ᵢ = 1/√N*∑ₖc†ₖ exp(-i*k*rᵢ) 

"""
struct PHVertexRepresentation{H<:RepresentationGenerator, Vq, Vk, T} <: MatrixRepresentation
    q::Vq
    k₁::Vk
    k₂::Vk
    table::T
    gauge::Symbol
    function PHVertexRepresentation{H}(q, k₁, k₂, table, gauge::Symbol=:icoordinate) where H<:RepresentationGenerator
        @assert gauge∈(:rcoordinate, :icoordinate) "PHVertexRepresentation error: gauge must be :rcoordinate or :icoordinate."
        return new{H, typeof(q), typeof(k₁), typeof(table)}(q, k₁, k₂, table, gauge)
    end
end
"""
    PHVertexRepresentation{H}(table, gauge::Symbol=:icoordinate)  where {H<:RepresentationGenerator}
    PHVertexRepresentation{H}(q, table, gauge::Symbol=:icoordinate) where {H<:RepresentationGenerator}

Get the matrix representation of particle-hole channel.
"""
@inline PHVertexRepresentation{H}(table, gauge::Symbol=:icoordinate) where {H<:RepresentationGenerator} = PHVertexRepresentation{H}(nothing, nothing, nothing, table, gauge)
@inline PHVertexRepresentation{H}(q, table, gauge::Symbol=:icoordinate) where {H<:RepresentationGenerator} = PHVertexRepresentation{H}(q, nothing, nothing, table, gauge)
@inline Base.valtype(phvr::PHVertexRepresentation) = valtype(typeof(phvr))
@inline Base.valtype(::Type{<:PHVertexRepresentation{H}}) where {H<:RepresentationGenerator} = Matrix{promote_type(valtype(eltype(H)), Complex{Int})}
@inline Base.valtype(R::Type{<:PHVertexRepresentation}, ::Type{<:Union{Operator, Operators}}) = valtype(R)
@inline Base.zero(mr::PHVertexRepresentation) = zeros(eltype(valtype(mr)), length(mr.table)^2, length(mr.table)^2)
@inline Base.zero(mr::PHVertexRepresentation, ::Union{Operator, Operators}) = zero(mr)
@inline (mr::PHVertexRepresentation)(m::Operator; kwargs...) = add!(zero(mr, m), mr, m; kwargs...)
"""
    add!(dest::AbstractMatrix, mr::PHVertexRepresentation{<:RepresentationGenerator}, m::Operator; kwargs...)

Get the matrix representation of an operator and add it to destination.
"""
function add!(dest::AbstractMatrix, mr::PHVertexRepresentation{<:RepresentationGenerator}, m::Operator; kwargs...)
    @assert rank(m) == 4 "add! error: rank(operator) $(rank(m)) == 4"
    n = length(mr.table)
    destr = zeros(eltype(dest), n, n, n, n)
    cr = Int[]
    an = Int[]
    for i = 1:rank(m)
        if m[i].index.iid.nambu == creation 
            push!(cr, i)
        else
            push!(an, i)
        end
    end
    p = [cr[1], an[1], cr[2], an[2]]
    sign = isevenperm(p) ? 1 : -1
    seq₁, seq₂, seq₃, seq₄ = mr.table[m[p[1]].index], mr.table[m[p[2]].index], mr.table[m[p[4]].index], mr.table[m[p[3]].index]
    
    if issamesite(m[p[1]], m[p[2]]) && issamesite(m[p[3]], m[p[4]]) && issamesite(m[p[2]], m[p[3]])
        destr[seq₁, seq₂, seq₃, seq₄] += m.value*sign
        destr[seq₄, seq₃, seq₂, seq₁] += m.value*sign
        destr[seq₁, seq₃, seq₂, seq₄] += -m.value*sign
        destr[seq₄, seq₂, seq₃, seq₁] += -m.value*sign
    elseif issamesite(m[p[1]], m[p[2]]) && issamesite(m[p[3]], m[p[4]])
        coordinate = mr.gauge==:rcoordinate ? m[p[3]].rcoordinate - m[p[1]].rcoordinate : m[p[3]].icoordinate - m[p[1]].icoordinate
        phaseq = isnothing(mr.q) ? one(eltype(dest)) : convert(eltype(dest), exp(-1im*dot(mr.q, coordinate)))
        destr[seq₁, seq₂, seq₃, seq₄] += m.value*sign*phaseq
        destr[seq₄, seq₃, seq₂, seq₁] += m.value*sign*conj(phaseq)
        if !(isnothing(mr.k₁)) && !(isnothing(mr.k₂)) #no test
            phasek = convert(eltype(dest), exp(-1im*dot(mr.k₂-mr.k₁, coordinate)))
            destr[seq₁, seq₃, seq₂, seq₄] += -m.value*sign*phasek
            destr[seq₄, seq₂, seq₃, seq₁] += -m.value*sign*conj(phasek)
        end
    elseif issamesite(m[p[1]], m[p[4]]) && issamesite(m[p[3]], m[p[2]])
        coordinate = mr.gauge==:rcoordinate ? m[p[3]].rcoordinate - m[p[1]].rcoordinate : m[p[3]].icoordinate - m[p[1]].icoordinate
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
    U::U
    function RPA(tba::AbstractTBA, U::RepresentationGenerator)
        new{typeof(tba), typeof(U)}(tba, U)
    end
end
@inline Parameters(rpa::RPA) = Parameters{(keys(Parameters(rpa.tba))...,keys(Parameters(rpa.U))...)}((Parameters(rpa.tba))...,(Parameters(rpa.U))... )
"""
    RPA(tba::AbstractTBA, uterms::Tuple{Vararg{Term}})
    RPA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, uterms::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)

Construct a `RPA` type.
"""
function RPA(tba::AbstractTBA, uterms::Tuple{Vararg{Term}})
    table = Table(tba.H.hilbert, OperatorUnitToTuple(:site, :orbital, :spin))
    u = OperatorGenerator(uterms, tba.H.bonds, tba.H.hilbert; half=false, table=table, boundary=tba.H.operators.boundary)
    return RPA(tba, u)
end
function RPA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, uterms::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)
    tba = TBA(lattice, hilbert, terms; neighbors=neighbors, boundary=boundary)
    table = Table(hilbert, OperatorUnitToTuple(:site, :orbital, :spin))
    u = OperatorGenerator(uterms, bonds(lattice, neighbors), hilbert; half=false, table=table, boundary=boundary)
    return RPA(tba, u)
end
@inline function update!(rpa::RPA; k=nothing, kwargs...)
    if length(kwargs)>0
        update!(rpa.U; kwargs...)
        update!(rpa.tba.H; kwargs...)
    end
    return rpa
end
"""
    matrix(rpa::RPA, field::Symbol=:U; k=nothing, gauge=:icoordinate, kwargs...) -> Matrix

Get matrix of particle-hole channel of interaction.
"""
@inline function matrix(rpa::RPA, field::Symbol=:U; k=nothing, gauge=:icoordinate, kwargs...)
    field == :U && return PHVertexRepresentation{typeof(rpa.U)}(k, rpa.U.table, gauge)(expand(rpa.U))
    field == :tba && return matrix(rpa.tba; k=k, gauge=gauge, kwargs...)
end

"""
    ParticleHoleSusceptibility{P<:Union{ReciprocalPath,ReciprocalZone}, RZ<:ReciprocalZone, E<:AbstractVector, S<:Operators} <: Action

Calculate the particle-hole susceptibility within random phase approximation.
Attribute `options` contains (η=0.01, gauge =:icoordinate, temperature=1e-12, μ=0.0, findk = false)
"""
struct ParticleHoleSusceptibility{P<:Union{ReciprocalPath,ReciprocalZone}, RZ<:ReciprocalZone, E<:AbstractVector, S<:Operators} <: Action
    path::P
    bz::RZ
    energies::E
    operators::Tuple{AbstractVector{S}, AbstractVector{S}}
    options::Dict{Symbol, Any}
    function ParticleHoleSusceptibility(path::Union{ReciprocalPath,ReciprocalZone}, bz::ReciprocalZone, energies::AbstractVector, operators::Tuple{AbstractVector{S}, AbstractVector{S}}, options::Dict{Symbol, Any}) where S<:Operators
        @assert keys(path)==(:k,) "ParticleHoleSusceptibility error: the name of the momenta in the path must be :k."
        @assert operators[1]|>length == length(operators[2]) "ParticleHoleSusceptibility error: the number of left operators must be equal to that of right operators."
        new{typeof(path), typeof(bz), typeof(energies), S}(path, bz, energies, operators, options)
    end
end
"""
    ParticleHoleSusceptibility(path::Union{ReciprocalPath,ReciprocalZone}, bz::ReciprocalZone, energies::AbstractVector, operators::Tuple{AbstractVector{<:Operators}, AbstractVector{<:Operators}}; options...)

Construct a `ParticleHoleSusceptibility` type.
"""
@inline function ParticleHoleSusceptibility(path::Union{ReciprocalPath,ReciprocalZone}, bz::ReciprocalZone, energies::AbstractVector, operators::Tuple{AbstractVector{<:Operators}, AbstractVector{<:Operators}}; options...)
    ParticleHoleSusceptibility(path, bz, energies, operators, convert(Dict{Symbol, Any}, options))
end

@inline function initialize(ins::ParticleHoleSusceptibility, rpa::RPA)
    x = collect(Float64, 0:(length(ins.path)-1))
    y = collect(Float64, ins.energies)
    z = zeros(ComplexF64, length(y), length(x))
    z₀ = zeros(ComplexF64, length(y), length(x))
    return (x, y, z, z₀)
end
function run!(rpa::Algorithm{<:RPA}, ins::Assignment{<:ParticleHoleSusceptibility})
    U, tba = rpa.frontend.U, rpa.frontend.tba
    n = length(U.table)
    nq = length(ins.action.path)
    vph = zeros(ComplexF64, n^2, n^2, nq)
    gauge = get(ins.action.options, :gauge, :icoordinate)
    for (i, q) in enumerate(ins.action.path)
        vph[:, :, i] = matrix(rpa.frontend, :U; k=q, gauge=gauge)
    end
    eta = get(ins.action.options, :η, 0.01)
    tem = get(ins.action.options, :temperature, 1e-12)
    mu = get(ins.action.options, :μ, 0.0)
    scflag = kind(tba) == Fermionic(:BdG) ? true : false
    kloc = get(ins.action.options, :findk, false)
    if kloc
        ndim, nk = dimension(tba), length(ins.action.bz)
        eigenvc = zeros(ComplexF64, ndim, ndim, nk)
        eigenval = zeros(Float64, ndim, nk)
        for (i, k) in enumerate(ins.action.bz)
            F = eigen(matrix(tba, k=k, gauge=:icoordinate)) 
            eigenval[:, i] = F.values
            eigenvc[:, :, i] = F.vectors
        end
        χ₀, χ = chiq(eigenvc, eigenval, ins.action.bz, ins.action.path, vph, ins.action.energies; eta=eta, tem=tem,mu=mu, scflag=scflag)
    else
        χ₀, χ = chiq(tba, ins.action.bz, ins.action.path, vph, ins.action.energies; eta=eta, tem=tem, mu=mu, scflag=scflag, gauge=gauge, ins.action.options...)
    end 
    savetag = get(ins.action.options, :save, false)
    filename = get(ins.action.options, :filename, "chi")
    savetag && (serialize(join([filename,"0"]), χ₀))
    if kloc == true 
        gauge₁ = :rcoordinate
    else
        gauge₁ = ( gauge == :rcoordinate ) ? :icoordinate : :rcoordinate
    end
    ins.data[3][:, :] = correlation(χ, ins.action.path, ins.action.operators, U.table; gauge=gauge₁)
    ins.data[4][:, :] = correlation(χ₀, ins.action.path, ins.action.operators, U.table; gauge=gauge₁)      
end
"""
    matrix!(m::Matrix{<:Number}, operators::Operators, table::Table, k; gauge=:rcoordinate)

Return the matrix representation of `operators`.
"""
function matrix!(m::Matrix{<:Number}, operators::Operators, table::Table, k; gauge=:rcoordinate)
    m[:, :] .= zero(eltype(m))
    for op in operators
        phase = gauge == :icoordinate ? one(eltype(m)) : convert(eltype(m), exp(1im*dot(k, op[1].rcoordinate - op[1].icoordinate))) 
        seq₁ = table[op[1].index']
        seq₂ = table[op[2].index]
        m[seq₁, seq₂] += op.value*phase
    end
    return m
end
"""
    correlation(χ::Array{<:Number, 4}, path::Union{ReciprocalPath, ReciprocalZone}, operators::Tuple{AbstractVector{S}, AbstractVector{S}}, table; gauge=:rcoordinate) where {S <: Operators} -> Matrix

Return physical particle-hole susceptibility.
"""
function correlation(χ::Array{<:Number, 4}, path::Union{ReciprocalPath, ReciprocalZone}, operators::Tuple{AbstractVector{S}, AbstractVector{S}}, table; gauge=:rcoordinate) where {S <: Operators}
    nop₁ = length(operators[1])
    @assert length(operators[1]) == length(operators[2]) "correlation error: the number of left operators must be equal to that of right operators."
    n = length(table)
    m₁ = zeros(ComplexF64, n, n)
    m₂ = zeros(ComplexF64, n, n) 
    nw, nq = size(χ, 3), size(χ, 4)
    χm = reshape(χ, n, n, n, n, nw, nq)
    res = zeros(eltype(χ), nw, nq)
    for (i, momentum) in enumerate(path)
        for i1 in 1:nop₁
            ops₁ = operators[1][i1]
            ops₂ = operators[2][i1]
            matrix!(m₁, ops₁, table, momentum; gauge=gauge)
            matrix!(m₂, ops₂, table, -momentum; gauge=gauge)   
            for a in 1:n
                for b in 1:n
                    for c in 1:n
                        for d in 1:n
                            res[:, i] += χm[a, b, c, d, :, i]*m₁[b, a]*m₂[c, d]
                        end
                    end
                end
            end
        end
    end
    return res 
end



"""
    fermifunc(e::T, temperature::T=1e-12, mu::T=0.0) where {T<:Real} -> Float64
 
Fermi distribution function.
"""
function fermifunc(e::T, temperature::T=1e-12, mu::T=0.0) where {T<:Real}
    if temperature > 1e-10
        f=(e-mu)/temperature
        if f>20
            f=0.0
        elseif f<-20
            f=1.0
        else
            f=1/(1+exp(f))
        end
    else
        f = e-mu
        if f>0.0
            f=0.0
        else
            f=1.0
        end
    end        
    return f
end

"""
    _chikq0(tba::AbstractTBA, k::AbstractVector, q::AbstractVector, omega::Float64; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, kwargs...)

Return chi0(k,q)_{ij,mn}= chi0(k,q)_{-+,+-}, chi0(k,q)_{12,34}== <c^\\dagger_{k,2}c_{k-q ,1}c^\\dagger_{k-q,3}c_{k,4}> 
"""
function _chikq0(tba::AbstractTBA, k::AbstractVector, q::AbstractVector, omega::Float64; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, kwargs...)
    Fk = eigen(Hermitian(matrix(tba; k = k, kwargs...)))
    Fkq = eigen(Hermitian(matrix(tba; k = k-q, kwargs...)))
    n = length(Fk.values)
    Ek = Fk.values
    Ekq = Fkq.values
    temp = diagm([-( fermifunc(Ek[i], tem, mu) - fermifunc(Ekq[j], tem, mu) ) / ( omega + eta*im + Ek[i] - Ekq[j] ) for i=1:n for j = 1:n])
    u = kron( conj(Fk.vectors), Fkq.vectors )
    return u*temp*adjoint(u)
end

function _chikqsc0(tba::AbstractTBA, k::AbstractVector, q::AbstractVector, omega::Float64; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, kwargs...)
    Fk = eigen(Hermitian(matrix(tba; k = k, kwargs...)))
    Fkq = eigen(Hermitian(matrix(tba; k = k-q, kwargs...)))
    n = length(Fk.values)
    @assert mod(n, 2) == 0 "chikqsc0 error: even dimension for Hamiltonian"
    m = n ÷ 2
    Ek = Fk.values
    Ekq = Fkq.values
    temp = diagm([ ( fermifunc(Ek[i], tem, mu) - fermifunc(Ekq[j], tem, mu) ) / ( omega + eta*im + Ek[i] - Ekq[j] ) for i=1:n for j = 1:n])
    uk = Fk.vectors[1:m, :]
    vk = Fk.vectors[m+1:n, :]
    ukq = Fkq.vectors[1:m, :]
    vkq = Fkq.vectors[m+1:n, :]
    u = kron(conj(uk), ukq)
    vtemp = kron(vk, conj(vkq))
    v = reshape( permutedims( reshape(vtemp, (m, m, n*n)), [3, 2, 1] ), (n*n, m*m))
    return -u*temp*adjoint(u) + u*temp*v
end

function _chikq0(Ek::Vector{Float64}, Ekq::Vector{Float64}, uk::Matrix{ComplexF64}, ukq::Matrix{ComplexF64}, omega::Float64; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0)
    n = length(Ek)
    temp = diagm([ -( fermifunc(Ek[i], tem, mu) - fermifunc(Ekq[j], tem, mu) ) / ( omega + eta*im + Ek[i] - Ekq[j] ) for i=1:n for j=1:n])
    u = kron( conj(uk), ukq )
    return u*temp*adjoint(u)
end
function _chikqsc0(Ek::Vector{Float64}, Ekq::Vector{Float64}, uk::Matrix{ComplexF64}, ukq::Matrix{ComplexF64}, omega::Float64; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0)
    n = length(Ek)
    temp = diagm([ (fermifunc(Ek[i], tem, mu) - fermifunc(Ekq[j], tem, mu)) / (omega + eta*im + Ek[i] - Ekq[j]) for i=1:n for j=1:n])
    m = Int( n/2 )
    uk0  = uk[1:m, :]
    vk0  = uk[m + 1:n, :]
    ukq0 = ukq[1:m, :]
    vkq0 = ukq[m + 1:n, :]
    u = kron(conj(uk0), ukq0)
    vtemp = kron(vk0, conj(vkq0))
    v = reshape( permutedims( reshape(vtemp, (m, m, n*n)), [3, 2, 1] ), (n*n, m*m))
    return -u*temp*adjoint(u) + u*temp*v
end

"""                
    chiq(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, vph::Array{T,3}, omegam::AbstractVector; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false, kwargs...) where T<:Number -> Tuple{ Array{ComplexF64, 4}, Array{ComplexF64, 4} }            

Get the particle-hole susceptibilities χ⁰(ω,q) and χ(ω,q). The spectrum function is satisfied by A(ω,q) = Im[χ(ω+i*0⁺,q)].

# Arguments
- `vph` is the bare particle-hole vertex (vph[ndim,ndim,nq])
- `omegam` store the energy points
- `eta` is the magnitude of broaden
- `tem` is the temperature
- `mu` is the chemical potential
- `scflag` == false (default, no superconductivity), or true ( BdG model)
- `kwargs` is transfered to `matrix(tba; kwargs...)` function
"""             
function chiq(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, vph::Array{T,3}, omegam::AbstractVector; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false, kwargs...) where T<:Number
    nk = length(bz)
    nq = length(path)
    nw = length(omegam)
    ndim = size(vph, 1)
    chi0 = SharedArray{ComplexF64}(ndim, ndim, nw, nq)
    chi = SharedArray{ComplexF64}(ndim, ndim, nw, nq)
    idmat = Matrix{Float64}(I, ndim, ndim)
    if scflag == false              
        @sync @distributed for j=1:nq*nw #iq=1:nq
            iq, iw = fldmod1(j, nw)
            q = path[iq]
            ω = omegam[iw]
            chi0[:, :, iw, iq] = @distributed (+) for i = 1:nk
                1/nk*_chikq0(tba, bz[i], q, ω; eta=eta, tem=tem, mu=mu, kwargs...)
            end
            chi[:, :, iw, iq] = chi0[:, :, iw, iq]*inv(idmat + vph[:, :, iq]*chi0[:, :, iw, iq])
        end
    else
        @sync @distributed for j = 1:nq*nw #iq=1:nq
            iq, iw = fldmod1(j, nw)
            q = path[iq]
            ω = omegam[iw]
            chi0[:, :, iw, iq] = @distributed (+) for i = 1:nk
                1/nk*_chikqsc0(tba, bz[i], q, ω; eta=eta, tem=tem, mu=mu, kwargs...)
            end
            chi[:, :, iw, iq] = chi0[:, :, iw, iq]*inv(idmat + vph[:, :, iq]*chi0[:, :, iw, iq])
        end
    end
    return Array(chi0), Array(chi)
end

"""
    findk(kq::AbstractVector, bz::ReciprocalZone) -> Int

Find the index of k point in the reduced Brillouin Zone `bz`, i.e. bz[result] ≈ kq
"""
function findk(kq::AbstractVector, bz::ReciprocalZone)
    @assert length(kq) == length(eltype(bz.reciprocals)) "findk error: dismatch of k-space dimension "
    n = length(bz.bounds)
    stepk = zeros(Float64, length(kq), n)
    nab = Int[]
    for (i, segment) in enumerate(bz.bounds)
        push!(nab, segment.length)
        step = segment[2] - segment[1]
        stepk[:, i] += bz.reciprocals[i]*step
    end
    orign = bz.momenta[1]
    x = stepk\(kq - orign)
    intx = round.(Int, x)
    res = reverse(map(mod, intx, nab) .+ 1)
    nabc = cumprod(reverse(nab))
    res0 = 0
    for i = n:-1:2
        res0 += (res[i] - 1)*nabc[i - 1]
    end
    res0 += res[1]
    return res0
end
"""
    chiq(eigenvc::Array{ComplexF64, 3}, eigenval::Array{Float64, 2}, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, vph::Array{T, 3}, omegam::Vector{Float64}; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false) where T<:Number -> Tuple{ Array{ComplexF64, 4}, Array{ComplexF64, 4} } 
    
"""     
function chiq(eigenvc::Array{ComplexF64,3}, eigenval::Array{Float64,2}, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, vph::Array{T,3}, omegam::AbstractVector; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false) where T<:Number
    nk = length(bz)
    nq = length(path)
    nw = length(omegam)
    ndim = size(vph, 1)
    chi0 = SharedArray{ComplexF64}(ndim, ndim, nw, nq)
    chi = SharedArray{ComplexF64}(ndim, ndim, nw, nq)
    idmat = Matrix{Float64}(I, ndim, ndim)
    if scflag == false
        @sync @distributed for j = 1:nq*nw 
        iq, iw = fldmod1(j, nw)
        q = path[iq]
        ω = omegam[iw]
        chi0[:,:,iw,iq] = @distributed (+) for i = 1:nk
            kq = bz[i] - q
            ikq = findk(kq, bz)
            1/nk*_chikq0(eigenval[:, i], eigenval[:, ikq], eigenvc[:, :, i], eigenvc[:, :, ikq], ω; eta=eta,tem=tem, mu=mu)
        end
            chi[:, :, iw, iq] = chi0[:, :, iw, iq]*inv(idmat + vph[:, :, iq]*chi0[:, :, iw, iq])
        end
    else
        @sync @distributed for j = 1:nq*nw 
        iq, iw = fldmod1(j,nw)
        q = path[iq]
        ω = omegam[iw]
        chi0[:, :, iw, iq] = @distributed (+) for i = 1:nk
            kq = bz[i] - q
            ikq = findk(kq, bz)
            1/nk*_chikqsc0(eigenval[:, i], eigenval[:, ikq], eigenvc[:, :, i], eigenvc[:, :, ikq], ω; eta=eta, tem=tem, mu=mu)
        end
            chi[:, :, iw, iq] = chi0[:, :, iw, iq]*inv(idmat + vph[:, :, iq]*chi0[:, :, iw, iq])
        end
    end
    return Array(chi0), Array(chi)
end

"""                
    chiq0(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, omegam::Vector{Float64}; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false, kwargs...) -> Array{Float64, 4}

Get the particle-hole susceptibilities χ⁰(ω,q).\\
`omegam` store the energy points;\\
`eta` is the magnitude of broaden;\\
`tem` is the temperature;\\
`mu` is the chemical potential.\\
'kwargs' is transfered to `matrix(tba;kwargs...)` function\\
`scflag` == false(default) => particle-hole channel, ==true => Nambu space.
"""  
function chiq0(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, omegam::AbstractVector; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false, kwargs...)
    nk = length(bz)
    nq = length(path)
    nw = length(omegam)
    ndim = dimension(tba)
    chi0 = zeros(ComplexF64, ndim^2, ndim^2, nw, nq)
    if scflag == false
        for iq = 1:nq
            for iw = 1:nw
                for i = 1:nk
                    chi0[:,:,iw,iq] += 1/nk*_chikq0(tba, bz[i], path[iq], omegam[iw]; eta=eta, tem=tem, mu=mu, kwargs...)
                end
            end
        end
    else
        for iq = 1:nq
            for iw = 1:nw
                for i = 1:nk
                    chi0[:, :, iw, iq] += 1/nk*_chikqsc0(tba, bz[i], path[iq], omegam[iw]; eta=eta, tem=tem, mu=mu, kwargs...)
                end
            end
        end   
    end
    return chi0
end
"""
    chiq(vph::Array{T, 3}, chi0::Array{ComplexF64, 4}) where T<:Number -> Array{ComplexF64, 4}

Get the susceptibility χ_{αβ}(ω,q).
"""
function chiq(vph::Array{T,3}, chi0::Array{ComplexF64,4}) where T<:Number
    nq = size(chi0, 4)
    nw = size(chi0, 3)
    ndim = size(vph, 1)                                
    chi = zeros(ComplexF64, ndim, ndim, nw, nq)
    idmat = Matrix{Float64}(I, ndim, ndim)
    for iq = 1:nq
        for iw = 1:nw
            chi[:, :, iw, iq] = chi0[:, :, iw, iq]*inv(idmat + vph[:, :, iq]*chi0[:, :, iw, iq])
        end
    end
    return chi
end         
                
# eigenproblem of RPA
"""
    EigenRPA{P<:Union{ReciprocalPath,ReciprocalZone}, RZ<:ReciprocalZone} <: Action

Eigenproblem for standard random phase approximation. 
"""
struct EigenRPA{P<:Union{ReciprocalPath,ReciprocalZone}, RZ<:ReciprocalZone} <: Action
    path::P
    bz::RZ
    onlyvalue::Bool
    options::Dict{Symbol, Any}
    function EigenRPA(path::Union{ReciprocalPath,ReciprocalZone}, bz::ReciprocalZone, onlyvalue::Bool, options::Dict{Symbol, Any}) 
        @assert keys(path) == (:k,) "ParticleHoleSusceptibility error: the name of the momenta in the path must be :k."
        new{typeof(path), typeof(bz)}(path, bz, onlyvalue, options)
    end
end
"""
    EigenRPA(path::Union{ReciprocalPath, ReciprocalZone}, bz::ReciprocalZone; onlyvalue::Bool=true,  options...)

Construct a `EigenRPA` type. Attribute `options` contains (gauge=:icoordinate, exchange=false, η=1e-8, temperature=1e-12, μ=0.0, bnd=nothing)
"""
@inline function EigenRPA(path::Union{ReciprocalPath,ReciprocalZone}, bz::ReciprocalZone;onlyvalue::Bool=true, options...)
    EigenRPA(path, bz, onlyvalue, convert(Dict{Symbol, Any}, options))
end

@inline function initialize(ins::EigenRPA, rpa::RPA)
    x = collect( Float64, 0:( length(ins.path) - 1 ) )
    y = Vector{ComplexF64}[]
    vecs = Matrix{ComplexF64}[]
    orb2band = Matrix{ComplexF64}[]
    return x, y, vecs, orb2band
end
function run!(rpa::Algorithm{<:RPA}, ins::Assignment{<:EigenRPA})
    U, tba = rpa.frontend.U, rpa.frontend.tba
    n = length(U.table)
    nq = length(ins.action.path)
    nk = length(ins.action.bz)
    vph = zeros(ComplexF64, n^2, n^2, nk, nk, nq)
    gauge = get(ins.action.options, :gauge, :icoordinate)
    exchange = get(ins.action.options, :exchange, false)
    if exchange  
        for (i, q) in enumerate(ins.action.path)
            for (l, k₂) in enumerate(ins.action.bz)
                for (j, k₁) in enumerate(ins.action.bz)
                    vph[:, :, j, l, i] = PHVertexRepresentation{typeof(U)}(q, k₁, k₂, U.table, gauge=gauge)(expand(U))
                end
            end  
        end 

    else
        for (i, q) in enumerate(ins.action.path)
            for j in 1:nk
                for l in 1:nk
                    vph[:, :, j, l, i] = matrix(rpa.frontend, :U; k=q, gauge=gauge)  
                end
            end  
        end  
    end
    η = get(ins.action.options, :η, 1e-8)
    tem = get(ins.action.options, :temperature, 1e-12)
    mu = get(ins.action.options, :μ, 0.0)
    bnd = get(ins.action.options, :bnd, nothing)
    
    values, vectors, orb2band = chikqm(tba, ins.action.bz, ins.action.path, vph; tem=tem, mu=mu, onlyvalue=ins.action.onlyvalue, η=η, bnd=bnd, gauge=gauge, ins.action.options...)
    push!(ins.data[2], values...)
    push!(ins.data[3], vectors...)
    push!(ins.data[4], orb2band...)
end

"""
    chikqm(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalZone, ReciprocalPath}, vph::Array{T,5}; tem::Float64=1e-12, mu::Float64=0.0, onlyvalue::Bool=true, η::Float64=1e-6, bnd::Union{UnitRange{Int}, StepRange{Int,Int}, Vector{Int}, Nothing}=nothing, kwargs...) where T<:Number -> Tuple{ Array{Array{ComplexF64, 1}, 1}, Array{Matrix{ComplexF64}, 1}, Array{Matrix{ComplexF64}, 1}}


Get the eigenvalues, eigenvectors, and unitary transformation (from orbital to band) of particle-hole susceptibilities χ_{αβ}(ω,k1,k2,q). Now only the zero-temperature case is supported.\\
`vph` store the particle-hole vertex,e.g. vph[ndim,ndim,nk,nk,nq] where sqrt(nidm) is the number of degrees of freedom in the unit cell,nk=length(bz), nq=length(path). \\
`tem` is the temperature;\\
`mu` is the chemical potential.\\
`onlyvalue` : only need the eigenvalues ( isval=true,zero temperature ), isval=false denotes that the cholesky method is used.\\
`η`:small number to advoid the semipositive Hamiltonian,i.e. Hamiltonian+diagm([η,η,...])\\
`bnd`:select the bands to calculate the χ₀\\
`kwargs` store the keys which are transfered to `matrix` function.
"""                     
function chikqm(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalZone, ReciprocalPath}, vph::Array{T,5}; tem::Float64=1e-12, mu::Float64=0.0, onlyvalue::Bool=true, η::Float64=1e-6, bnd::Union{UnitRange{Int}, StepRange{Int, Int}, Vector{Int}, Nothing}=nothing, kwargs...) where T<:Number
    nk = length(bz)
    nq = length(path)
    val = Array{ComplexF64,1}[]
    vec = Array{ComplexF64,2}[]
    f2c = Array{ComplexF64,2}[]
    
    for iq = 1:nq
        chi0inv = Float64[]
        um = Array{ComplexF64,2}[]
        list = Int[1]
        g1 = Float64[]
        for i = 1:nk
            temp0, tempu, g0 = _chikq03(tba, bz[i], path[iq], 0.0; tem=tem, mu=mu, bnd=bnd, kwargs...)                          
            push!(um, tempu)
            push!(chi0inv, temp0...)
            push!(g1, g0...)
            n0 = size(tempu, 2)
            push!(list, n0)
        end 
        list0 = cumsum(list)
        wk = zeros(ComplexF64, list0[end]-1, list0[end]-1)
        for i = 1:nk
            for j = 1:nk
                wk[list0[j]:list0[j+1]-1, list0[i]:list0[i+1]-1] = transpose(conj(um[j]))*vph[:, :, j, i, iq]*um[i]/nk
            end
        end
        temp1 = wk       
        temp3 = Hermitian(-diagm(chi0inv)*1.0 + 1.0*temp1, :U)

        g = diagm(g1)
        if onlyvalue 
            temp2 = eigen(temp3*g)
        else
            nd = size(temp3, 1)
            ck = cholesky(temp3 + diagm([η for i=1:nd]),  NoPivot(); check = true)
            temp2 = eigen(ck.U*g*ck.L)
            temp4 = inv(ck.U)*temp2.vectors*diagm(sqrt.(abs.(temp2.values)))
            push!(vec, temp4)  
        end
        push!(val, temp2.values)
        push!(f2c, hcat(um...))
    end
    return val, vec, f2c
end
            
function _chikq03(tba::AbstractTBA, k::AbstractVector, q::AbstractVector, omega::Float64=0.0;tem::Float64=1e-12, mu::Float64=0.0, bnd::Union{UnitRange{Int}, StepRange{Int, Int}, Vector{Int}, Nothing}=nothing, kwargs...)
    @assert tem < 1e-10 "chikq03 error: now only support the zero temperature (<1e-10) $(tem). "
    Fk = eigen(Hermitian(matrix(tba; k=k, kwargs...)))
    Fkq = eigen(Hermitian(matrix(tba; k=k-q, kwargs...)))
    n = length(Fk.values)
    if isnothing(bnd)
        bnd0 = 1:n
    else
        bnd0 = bnd
    end
    Ek = Fk.values
    Ekq = Fkq.values
    ee1 = Float64[]
    ee2 = Float64[]
    lit1 = Int[]
    lit2 = Int[]
    one1 = Float64[]
    one2 = Float64[]
    for i = bnd0
        for j = bnd0
            ft = (fermifunc(Ek[i], tem, mu) - fermifunc(Ekq[j], tem, mu))
            if abs(ft) > eps()
                if Ek[i] - Ekq[j] < 0.0
                    push!(lit1, (i-1)*n + j )
                    push!(ee1, (omega + Ek[i] - Ekq[j])/ft)
                    push!(one1, 1.0)
                else Ek[i] - Ekq[j] > 0.0
                    push!(lit2, (i - 1)*n + j)
                    push!(ee2, (omega + Ek[i] - Ekq[j])/ft)
                    push!(one2, -1.0)
                end                           
            end
        end
    end                                                                    
    u = kron(conj(Fk.vectors), Fkq.vectors)   
    ee = vcat(ee1, ee2)
    uu = zeros(ComplexF64, n*n, length(ee))
    uu[:, 1:length(ee1)] = u[:, lit1]
    uu[:, length(ee1)+1:end] = u[:, lit2]
    return ee, uu, vcat(one1, one2)
end
"""
    projchi(val::Array{Array{T, 1}, 1}, vec::Array{Array{T1, 2}, 1}, f2c::Array{ Array{ComplexF64, 2}, 1}, omegam::Vector{Float64}, eta::Float64=1e-2) where {T<:Number,T1<:Number} -> Array{ComplexF64,4}

Get the ``\\chi_{ij,nm}(\\omega,q)```. The eigenvalues, eigenvectors, and unitary matrix are obtained by method `chikqm`.
"""
function projchi(val::Array{Array{T,1},1}, vec::Array{Array{T1, 2}, 1}, f2c::Array{ Array{ComplexF64, 2}, 1 }, omegam::AbstractVector; eta::Float64=1e-2) where {T<:Number, T1<:Number}
    nq = length(val)
    nw = length(omegam)
    ndim, n = size(f2c[1])
    chi = zeros(ComplexF64, ndim, ndim, nw, nq)
    for iq = 1:nq
        for iw = 1:nw
            tm = vec[iq]
            tmu = f2c[iq]*tm
            chi[:, :, iw, iq] = tmu*diagm([ 1/(-omegam[iw] - im*eta + val[iq][i]) for i=1:n ])*(diagm(sign.(val[iq])))*adjoint(tmu)
        end
    end
    return chi
end

 """
    projchiim(val::Array{Array{T, 1}, 1}, vec::Array{Array{T1, 2}, 1}, f2c::Array{ Array{ComplexF64, 2}, 1},omegam::Vector{Float64}, eta::Float64=1e-2) where {T<:Number, T1<:Number} -> Array{ComplexF64, 4}

Get the imaginary party of ``\\chi_{ij,nm}(\\omega,q)```. The eigenvalues,eigenvectors, and unitary matrix are obtained by method `chikqm`.
"""
function projchiim(val::Array{ Array{T, 1}, 1 }, vec::Array{ Array{T1, 2}, 1 }, f2c::Array{ Array{ComplexF64, 2}, 1}, omegam::AbstractVector; eta::Float64=1e-2) where {T<:Number, T1<:Number}
    nq = length(val)
    nw = length(omegam)
    ndim = size(f2c[1], 1)
    chi = zeros(ComplexF64, ndim, ndim, nw, nq)   
    for iq = 1:nq
        for iw = 1:nw
            for i = 1:length(val[iq])
                temp = f2c[iq]*vec[iq][:, i]
                chi[:, :, iw, iq] += temp*eta/((omegam[iw] - val[iq][i])^2 + eta^2)*adjoint(temp) 
            end
        end
    end
    return chi
end         

function save(filename::AbstractString, data::Array{<:Number,4})
    open(filename, "w") do f
        writedlm(f, data)
    end

end
"""
    @recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}}, mode::Symbol=:χ)

Define the recipe for the visualization of particle-hole susceptibilities.
"""
@recipe function plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}},mode::Symbol=:χ)
    title --> nameof(pack[1], pack[2])
    titlefontsize --> 10
    legend --> true
    @assert mode == :χ || mode == :χ0 "plot error: mode ∈ (:χ, :χ0)" 
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
    @recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:EigenRPA}})

Define the recipe for the visualization of particle-hole susceptibilities.
"""
@recipe function plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:EigenRPA}}, reim::Symbol=:real)
    title --> nameof(pack[1], pack[2])
    titlefontsize --> 10
    legend --> false
    n = length(pack[2].data[2][end])
    nq = length(pack[2].data[2])
    rpa_val = zeros(ComplexF64, nq, n)
    for (i, v) in enumerate(pack[2].data[2])
        rpa_val[i, :] =  v
    end  
    values = reim ==:real ? real.(rpa_val) : imag.(rpa_val)
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
"""
    @recipe plot(rz::ReciprocalZone, path::Union{ReciprocalPath,Nothing}=nothing)

Define the recipe for the visualization of a reciprocal zone and a reciprocal path
"""
@recipe function plot(rz::ReciprocalZone, path::Union{ReciprocalPath,Nothing}=nothing)
    title := "ReciprocalZone"
    titlefontsize --> 10
    legend := false
    aspect_ratio := :equal
    @series begin
        seriestype := :scatter
        coordinates = NTuple{length(eltype(rz)), eltype(eltype(rz))}[]
        for i in rz
            push!(coordinates, Tuple(i))
        end
        coordinates
    end
    if !(isnothing(path))
        coordinates = NTuple{length(eltype(path)), eltype(eltype(path))}[]
        for i in path
            push!(coordinates, Tuple(i))
        end  
        @series begin
            seriestype := :scatter
            coordinates
        end
        @series begin
            coordinates
        end
    end
end
# select a path from the ReciprocalZone
using QuantumLattices: atol, rtol, isonline
using Base.Iterators: product
using LinearAlgebra: norm
"""
    selectpath(stsp::Tuple{<:AbstractVector, <:AbstractVector}, rz::ReciprocalZone; ends::Tuple{Bool, Bool}=(true, false), atol::Real=atol, rtol::Real=rtol) -> Tuple(Vector{Vector{Float64}}, Vector{Int})

Select a path from the reciprocal zone.
"""
function selectpath(stsp::Tuple{<:AbstractVector, <:AbstractVector}, rz::ReciprocalZone; ends::Tuple{Bool, Bool}=(true, false), atol::Real=atol, rtol::Real=rtol) 
    start, stop = stsp[1], stsp[2]
    dimₖ = length(rz[1])
    dim₁ = length(rz.reciprocals)
    @assert dimₖ == dim₁ "selectpath error: the dimension of k point does not match with the the number of reciprocals"
    recpr = zeros(Float64, dimₖ, dim₁)
    for i in 1:dim₁
        recpr[:, i] = rz.reciprocals[i]
    end
    startrp = recpr\(start - rz[1]) 
    stoprp = recpr\(stop - rz[1]) 
    intstart = floor.(Int, round.(startrp; digits=4))
    intstop = floor.(Int, round.(stoprp; digits=4))
    mes = []
    for i in 1:dimₖ
        if intstop[i] >= intstart[i]
            step = 1
        elseif intstop[i] < intstart[i]
            step = -1
        end
        push!(mes, intstart[i]:step:intstop[i])
    end
    disps = [recpr*[disp...] for disp in product(mes...)]
    psegments = Vector{Float64}[]
    isegments, dsegments = Int[], []
    for (pos, k) in enumerate(rz)
        for disp in disps
            rcoord = k + disp
            if isonline(rcoord, start, stop; ends=ends, atol=atol, rtol=rtol)
                push!(psegments, rcoord)
                push!(isegments, pos)
                push!(dsegments, norm(rcoord-start))
            end
        end
    end
    p = sortperm(dsegments)
    return psegments[p], isegments[p]
end
"""
    selectpath(path::AbstractVector{<:Tuple{<:AbstractVector, <:AbstractVector}}, bz::ReciprocalZone;ends::Union{<:AbstractVector{Bool},Nothing}=nothing, atol::Real=atol, rtol::Real=rtol) -> Tuple(Vector{Vector{Float64}}, Vector{Int})  -> -> Tuple(ReciprocalPath, Vector{Int})

Select a path from the reciprocal zone. Return ReciprocalPath and positions of points in the reciprocal zone.
"""
function selectpath(path::AbstractVector{<:Tuple{<:AbstractVector, <:AbstractVector}}, bz::ReciprocalZone;ends::Union{<:AbstractVector{Bool},Nothing}=nothing, atol::Real=atol, rtol::Real=rtol)
    points = Vector{Float64}[]
    positions = Int[]
    endss = isnothing(ends) ? [false for _ in 1:length(path)] : ends 
    @assert length(path) == length(endss) "selectpath error: the number of ends is not equal to that of path."
    for (i,stsp) in enumerate(path)
        psegments, isegments = selectpath(stsp, bz; ends=(true, endss[i]), atol=atol, rtol=rtol)
        if i>1 && length(isegments)>0 && length(positions)>0 && positions[end] == isegments[1]
            popfirst!(psegments) 
            popfirst!(isegments) 
        end
            append!(points, psegments)
            append!(positions, isegments)
    end
    return ReciprocalPath(points), positions
end
end # module

