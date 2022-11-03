using QuantumLattices: Lattice, Hilbert, Fock, Hubbard, InterOrbitalInterSpin, InterOrbitalIntraSpin, SpinFlip, PairHopping
using QuantumLattices: OperatorGenerator, plain, OperatorUnitToTuple, Table, bonds, expand
using QuantumLattices: CompositeIndex, Index, FID, Parameters
using QuantumLattices: MatrixCoupling, @σ_str, Coulomb, Hopping, Segment, ReciprocalZone, ReciprocalPath, Onsite, Algorithm
using QuantumLattices: icoordinate, update!, reciprocals
using LinearAlgebra: dot
using RandomPhaseApproximation
using Plots: plot, plot!, savefig
using Test

@testset begin
    @test isevenperm([3,1,2,4]) == true
      op₁ = CompositeIndex(Index(1, FID(1,1, 1)), [0.0, -0.0], [0.0, 0.0])
      op₂ = CompositeIndex(Index(1, FID(1,1, 2)), [0.0, -0.0], [0.0, 0.0])
     @test issamesite(op₁, op₂) == true
 end


lattice = Lattice(
    [0.0, 0.0], [1/(sqrt(3)),0.0];
    vectors = [[sqrt(3)/2, -0.50], [0.0, 1.0]],
    )
b₁, b₂ = reciprocals(lattice)
hilbert = Hilbert(pid=>Fock{:f}(1, 2) for pid in 1:length(lattice))
UU = 1.0
U = Hubbard(:U, UU)
mx = MatrixCoupling(:, FID, :, σ"x", :)
my = MatrixCoupling(:, FID, :, σ"y", :)
mz = MatrixCoupling(:, FID, :, σ"z", :)
J1 = Coulomb(:J1, 0.5, 1, 1//4*(mx*mx + my*my + mz*mz);
        ishermitian=true,
        amplitude=nothing
    )
@testset "PHVertexRepresentation" begin 
    hilbert₀ = Hilbert(pid=>Fock{:f}(2, 2) for pid in 1:length(lattice))

    Jₕ = 0.2*UU 
    U′ = InterOrbitalInterSpin(Symbol("U′"), UU-2*Jₕ)
    UmJ = InterOrbitalIntraSpin(Symbol("U′-J"), UU-3*Jₕ)
    J = SpinFlip(Symbol("J"), Jₕ)
    Jp = PairHopping(:Jp, Jₕ)

    table = Table(hilbert₀, OperatorUnitToTuple(:site, :orbital, :spin))
    gen = OperatorGenerator((U, U′, UmJ, J, Jp, J1), bonds(lattice, 1), hilbert₀; half=false, table=table, boundary=plain);
    
    δ = [icoordinate(bond) for bond in bonds(lattice, 1) if bond.kind == 1]
    k = [0.1, 0.2]
    mat = PHVertexRepresentation{typeof(gen)}(k, table)(expand(gen))
    mat₂ = reshape(mat, 8, 8, 8, 8)
    seq₁, seq₂ = table[(1, 1, 1//2)], table[(1, 1, -1//2)]
    seq₃, seq₄ = table[(2, 1, 1//2)], table[(2, 1, -1//2)]
    @test mat₂[seq₁, seq₁, seq₄, seq₄] == sum( x->-0.5*exp(-1im*dot(k, x))/4, -δ )
    @test mat₂[seq₄, seq₃, seq₂, seq₁] == sum( x->0.5*exp(-1im*dot(k, x))/2, δ )
    @test mat₂[seq₄, seq₄, seq₃, seq₃] == Complex(1.0)
    seq₅, seq₆ = table[(1, 2, 1//2)], table[(1, 2, -1//2)]
    @test mat₂[seq₁,seq₂,seq₅,seq₆] == Complex(-0.2)
end

t1 = Hopping(:t1, -0.1580, 1)
mzz = Onsite(:m0, 0.1910, mz; amplitude=x->x[1].site== 1 ? 1 : -1)
rpa = RPA(lattice, hilbert, (t1, mzz), (U, J1); neighbors=1)
update!(rpa; U=0.2*0.59888, J1=0.39925*0.8)

sx = expand(Onsite(:sx, 1.0+0.0im, 1/2*mx+0.5im*my), bonds(lattice, 2), hilbert, half=false) 
sy = expand(Onsite(:sy, 1.0+0.0im, 1/2*mx-0.5im*my), bonds(lattice, 2), hilbert, half=false) 
sz = expand(Onsite(:sz, 1.0+0.0im, 1/2*mz), bonds(lattice, 2), hilbert, half=false) 

nx, ny= 6, 6
bz = ReciprocalZone(reciprocals(lattice), Segment(0, 1, nx), Segment(0//2, 2//2, ny))
path, = selectpath([(b₁*0, b₁), (b₁, b₂-b₁)],
    bz;
    ends=[false, true]
)
omegam = range(0.0, 1.20, length=201)
phs = ParticleHoleSusceptibility(path, bz, omegam, ([sx], [sy]);
    η=0.02,
    gauge=:rcoordinate, 
    save=false
    )
antirpa = Algorithm(:rpa, rpa );
tes = antirpa(:AFM₀, phs);

phs₁ = ParticleHoleSusceptibility(path, bz, omegam, ([sx], [sy]);
    η=0.02, 
    gauge=:icoordinate, 
    save=false
    )
phs₂ = ParticleHoleSusceptibility(path, bz, omegam, ([sx], [sy]);
    η=0.02, 
    findk=true, 
    save=false
    )
tes1 = antirpa(:AFM₁, phs₁);
tes2 = antirpa(:AFM₂, phs₂);
eigenrpa = EigenRPA(path, bz; onlyvalue=false, gauge=:rcoordinate)
tes3 = antirpa(:EigenRPA, eigenrpa);

@testset "rpa" begin
    matdata = tes[2].data[3] - tes1[2].data[3]
    @test findmax(abs, matdata)[1] < 1e-9
    @test findmax(abs, tes[2].data[3] - tes2[2].data[3])[1] < 1e-9
    @test isapprox( tes3[2].data[2][1][length(tes3[2].data[2][1])÷2 + 1] , 0.001684994305; atol=1e-9)
    plt = plot(tes1, clim=(0,10))
    plt₁ = plot(tes1, :χ0, clim=(0,5))
    plt₂ = plot(tes3)
    display(plt)
    display(plt₁)
    display(plt₂)
    savefig(plt, "χ.png")
    savefig(plt₁, "χ₀.png")
    savefig(plt₂, "eigen.png")
end