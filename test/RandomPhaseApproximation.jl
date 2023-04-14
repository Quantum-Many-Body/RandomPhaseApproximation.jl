using LinearAlgebra: dot
using Plots: plot, plot!, savefig
using QuantumLattices: Algorithm, BrillouinZone, CompositeIndex, Coulomb, Fock, FID, Hilbert, Hopping, Hubbard, Index, InterOrbitalInterSpin, InterOrbitalIntraSpin, Lattice, MatrixCoupling, OperatorGenerator, OperatorUnitToTuple, Onsite, PairHopping, Parameters, ReciprocalZone, SpinFlip, Table
using QuantumLattices: bonds, expand, icoordinate, reciprocals, selectpath, update!, @σ_str
using RandomPhaseApproximation
using RandomPhaseApproximation: fermifunc, isevenperm, issamesite
using TightBindingApproximation: Fermionic, TBA

import QuantumLattices: dimension

@testset "utilities" begin
    @test isevenperm([3, 1, 2, 4]) == true
    op₁ = CompositeIndex(Index(1, FID(1, 1, 1)), [0.0, -0.0], [0.0, 0.0])
    op₂ = CompositeIndex(Index(1, FID(1, 1, 2)), [0.0, -0.0], [0.0, 0.0])
    @test issamesite(op₁, op₂) == true

    @test fermifunc(1.0, 0.0, 0.0) == 0
    @test fermifunc(-1.0, 0.0, 0.0) == 1
    @test fermifunc(3, 0.1, 0.0) == 0
    @test fermifunc(-3, 0.1, 0.0) == 1
    @test fermifunc(0.1, 0.1, 0.0) == 1.0/(1+exp(1))
 end

@testset "PHVertexMatrix" begin 
    lattice = Lattice([0.0, 0.0], [1/(sqrt(3)),0.0]; vectors = [[sqrt(3)/2, -0.5], [0.0, 1.0]])
    hilbert = Hilbert(site=>Fock{:f}(2, 2) for site in 1:length(lattice))
    u = 1.0
    jₕ = 0.2*u
    U = Hubbard(:U, u)
    U′ = InterOrbitalInterSpin(Symbol("U′"), u-2*jₕ)
    UmJ = InterOrbitalIntraSpin(Symbol("U′-J"), u-3*jₕ)
    J = SpinFlip(Symbol("J"), jₕ)
    Jp = PairHopping(:Jp, jₕ)
    J₁ = Coulomb(:J₁, 0.5, 1, 1//4*(MatrixCoupling(:, FID, :, σ"x", :)^2+MatrixCoupling(:, FID, :, σ"y", :)^2+MatrixCoupling(:, FID, :, σ"z", :)^2))
    table = Table(hilbert, OperatorUnitToTuple(:site, :orbital, :spin))
    gen = OperatorGenerator((U, U′, UmJ, J, Jp, J₁), bonds(lattice, 1), hilbert; half=false, table=table)

    δ = [icoordinate(bond) for bond in bonds(lattice, 1) if bond.kind==1]
    k = [0.1, 0.2]
    mat = reshape(PHVertexMatrix{valtype(eltype(gen))}(k, table)(expand(gen)), 8, 8, 8, 8)
    seq₁, seq₂ = table[(1, 1, 1//2)], table[(1, 1, -1//2)]
    seq₃, seq₄ = table[(2, 1, 1//2)], table[(2, 1, -1//2)]
    @test mat[seq₁, seq₁, seq₄, seq₄] == sum(x->-0.5*exp(-1im*dot(k, x))/4, -δ)
    @test mat[seq₄, seq₃, seq₂, seq₁] == sum(x->0.5*exp(-1im*dot(k, x))/2, δ)
    @test mat[seq₄, seq₄, seq₃, seq₃] == Complex(1.0)
    seq₅, seq₆ = table[(1, 2, 1//2)], table[(1, 2, -1//2)]
    @test mat[seq₁, seq₂, seq₅, seq₆] == Complex(-0.2)
end

@testset "RPA" begin
    lattice = Lattice([0.0, 0.0], [1/(sqrt(3)),0.0]; vectors = [[sqrt(3)/2, -0.5], [0.0, 1.0]])
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in 1:length(lattice))
    t = Hopping(:t, -0.1580, 1)
    Δ = Onsite(:Δ, 0.1910, MatrixCoupling(:, FID, :, σ"z", :); amplitude=bond->isodd(bond[1].site) ? 1 : -1)
    U = Hubbard(:U, 1.0)
    J = Coulomb(:J, 0.5, 1, 1//4*(MatrixCoupling(:, FID, :, σ"x", :)^2+MatrixCoupling(:, FID, :, σ"y", :)^2+MatrixCoupling(:, FID, :, σ"z", :)^2))
    rpa = Algorithm(:rpa, RPA(lattice, hilbert, (t, Δ), (U, J); neighbors=1))
    @test Parameters(rpa.frontend) == (t=-0.1580, Δ=0.1910, U=1.0, J=0.5)

    update!(rpa; U=0.2*0.59888, J=0.39925*0.8)
    s⁺ = expand(Onsite(:s⁺, 1.0, MatrixCoupling(:, FID, :, σ"+", :)), bonds(lattice, 0), hilbert)
    s⁻ = expand(Onsite(:s⁻, 1.0, MatrixCoupling(:, FID, :, σ"-", :)), bonds(lattice, 0), hilbert)

    brillouinzone = BrillouinZone{:k}(reciprocals(lattice), 6)
    path = selectpath(brillouinzone, (0, 0)=>(1, 0), (1, 0)=>(-1, 1); ends=((true, false), (true, true)))[1]
    energies = range(0.0, 1.20, length=201)

    result₀ = rpa(:AFM₀, ParticleHoleSusceptibility(path, brillouinzone, energies, ([s⁺], [s⁻]); η=0.02, gauge=:rcoordinate, save=false))
    result₁ = rpa(:AFM₁, ParticleHoleSusceptibility(path, brillouinzone, energies, ([s⁺], [s⁻]); η=0.02, gauge=:icoordinate, save=false))
    result₂ = rpa(:AFM₂, ParticleHoleSusceptibility(path, brillouinzone, energies, ([s⁺], [s⁻]); η=0.02, findk=true, save=false))
    @test findmax(abs, result₀[2].data[3]-result₁[2].data[3])[1] < 1e-9
    @test findmax(abs, result₀[2].data[4]-result₁[2].data[4])[1] < 1e-9
    @test findmax(abs, result₀[2].data[3]-result₂[2].data[3])[1] < 1e-9
    @test findmax(abs, result₀[2].data[4]-result₂[2].data[4])[1] < 1e-9
    savefig(plot(result₀, :χ, clim=(0, 10)), "χ.png")
    savefig(plot(result₀, :χ0, clim=(0, 5)), "χ₀.png")

    χ₀ = chiq0(rpa.frontend.tba, brillouinzone, path, energies, η=0.02)
    χ₁ = chiq(vertex(rpa.frontend, path), χ₀)
    @test findmax(abs, result₀[2].data[3]-correlation(χ₁, path, ([s⁺], [s⁻]), rpa.frontend.interactions.table))[1] < 1e-9
    @test findmax(abs, result₀[2].data[4]-correlation(χ₀, path, ([s⁺], [s⁻]), rpa.frontend.interactions.table))[1] < 1e-9

    result = rpa(:EigenRPA, EigenRPA(path, brillouinzone; eigvals_only=false, gauge=:rcoordinate))
    @test isapprox(result[2].data[2][1][length(result[2].data[2][1])÷2+1], 0.001684994305; atol=1e-9)
    savefig(plot(result), "eigen.png")

    # χ₂ = chiq(result[2].data[2], result[2].data[3], result[2].data[3], energies; η=0.02)
    # @test findmax(abs, χ₂-χ₁)[1] < 1e-9
    # χ₃ = chiq(result[2].data[2], result[2].data[3], result[2].data[3], energies; η=0.02, imag_only=true)
    # @test findmax(abs, χ₃-χ₁)[1] < 1e-9
end

function hamiltonian(t::Float64, Δ::Float64; k=nothing, kwargs...) 
    @assert !isnothing(k) "hamiltonian error"
    ek = 2t * (cos(k[1])+cos(k[2]))
    dk = 2Δ * (cos(k[1])-cos(k[2]))
    return [ek  0   0   dk;
            0   ek  -dk 0;
            0   -dk -ek 0;
            dk  0   0   -ek
    ]
end
@inline dimension(::typeof(hamiltonian)) = 4

@testset "SC" begin
    lattice = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in 1:length(lattice))
    parameters = Parameters{(:t, :Δ)}(0.4, 0.4*0.299)
    rpa = Algorithm(:dx²y², RPA(TBA{Fermionic{:BdG}}(lattice, hamiltonian, parameters), hilbert, (Hubbard(:U, 0.4),); neighbors=1))
    brillouinzone = BrillouinZone(reciprocals(lattice), 16)
    reciprocalzone = ReciprocalZone(reciprocals(lattice), -1=>1, -1=>1; length=33, ends=(true, true))
    s⁺ = expand(Onsite(:s⁺, 1.0, MatrixCoupling(:, FID, :, σ"+", :)), bonds(lattice, 0), hilbert)
    s⁻ = expand(Onsite(:s⁻, 1.0, MatrixCoupling(:, FID, :, σ"-", :)), bonds(lattice, 0), hilbert)
    result = rpa(:χ⁺⁻, ParticleHoleSusceptibility(reciprocalzone, brillouinzone, range(0.0, 0.0, length=1), ([s⁺], [s⁻]); η=0.02,findk=true, save=false))
    savefig(plot(result, 0.0, 1e-3, :χ, :re), "dx²y²-re.png")
    savefig(plot(result, 0.0, 1e-3, :χ, :im), "dx²y²-im.png")
end