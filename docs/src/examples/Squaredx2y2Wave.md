```@meta
CurrentModule = RandomPhaseApproximation
```

# dx2y2-wave superconductor
Spin excitation of t + Δ + Hubbard model on square lattice.

## Spectrum of spin excitation

The following codes could compute the spin excitation within random phase approximation.

```@example BdG
using QuantumLattices
using RandomPhaseApproximation
using Plots
using TightBindingApproximation: EnergyBands

lattice = Lattice(
    [0.0, 0.0];
    vectors=[[1.0, 0.0], [0.0, 1.0]]
)
b₁, b₂ = lattice.reciprocals[1], lattice.reciprocals[2]
hilbert = Hilbert(pid=>Fock{:f}(1, 2) for pid in 1:length(lattice))

pair = Coupling(Index(:, FID(:, 1//2, :)), Index(:, FID(:, -1//2, :))) -  Coupling(Index(:, FID(:, -1//2, :)), Index(:, FID(:, 1//2, :)))

t1 = Hopping(:t1, 1.0*0.4, 1)
Δ = Pairing(
    :dx2y2, 
    0.299*0.4, 
    1, 
    pair;
    amplitude=x->(
        ϕ=azimuth(rcoordinate(x));
        any(≈(ϕ),(0,π)) && return x[1].site == 1 && x[2].site==2 ? 1 : -1 ;
        any(≈(ϕ),(π/2, 3π/2)) && return x[1].site == 1 && x[2].site==2 ? -1 : 1 
    )       
 )

U = Hubbard(:U, 0.4)
mx = MatrixCoupling(:, FID, :, σ"x", :)
my = MatrixCoupling(:, FID, :, σ"y", :)
mz = MatrixCoupling(:, FID, :, σ"z", :)

#define the RPA frontend 
rpa = RPA(lattice, hilbert, (t1, Δ), (U, ); neighbors=1)

#plot energy bands
pathek = ReciprocalPath{:k}(
    lattice.reciprocals, 
    (0//2, 0//2)=>(1//2, 0//2), 
    (1//2, 0//2)=>(1//2, 1//2), 
    (1//2, 1//2)=>(0, 0),
    length=50
)
etba = Algorithm(:SquareAFM, rpa.tba)(:eband, EnergyBands(pathek, gauge=:rcoordinate))

plt1 = plot(
    etba, 
    xticks=(
        [0, 50, 100, 150], 
        ["(0,0)", "(π,0)","(π,π)","(0,0)"]
    )
)
#display(plt1)
```

The transverse spin excitation is calculated.
```@example BdG
nx, ny= 16, 16
bz = ReciprocalZone(lattice.reciprocals, 
    Segment(0, 1, nx), 
    Segment(0//2, 2//2, ny)
)
path, = selectpath(
    [(b₁*0, b₁/2), (b₁/2, b₂/2+b₁/2), (b₂/2+b₁/2, b₂*0)],
    bz;
    ends=[false, false, true]
)
s⁺ = expand(Onsite(:sx, 1.0+0.0im, 1/2*mx+0.5im*my), bonds(lattice, 1), hilbert, half=false) 
s⁻ = expand(Onsite(:sy, 1.0+0.0im, 1/2*mx-0.5im*my), bonds(lattice, 1), hilbert, half=false) 
sz = expand(Onsite(:sz, 1.0+0.0im, 1/2*mz), bonds(lattice, 1), hilbert, half=false) 

phs = ParticleHoleSusceptibility(
    path, 
    bz, 
    range(0.0, 4.0, length=400), 
    ([s⁺], [s⁻]);
    η=0.02,
    save=false,
    findk=true
)
antirpa = Algorithm(:dAFM, rpa);
tespm = antirpa(:chipm, phs);

#plot spectrum of longitudinal spin excitation
plt = plot(
    tespm, 
    xticks=(
        [0, 16/2, 32/2, 48/2, 64/2+1],
        ["(0,0)",  "(π,0)","(π,π)","(0,0)"]
    ),
    clims=(0, 0.5)
)
#display(plt)
```

Another way to define the TBA.
```@example BdG
import QuantumLattices: dimension
table = Table(hilbert, OperatorUnitToTuple(:site, :orbital, :spin))
function hamiltonian(t::Float64, delta::Float64;k=nothing, kwargs...) 
    @assert !isnothing(k) "hamiltonian error"
    ek = 2*t*(cos(k[1]) + cos(k[2]))
    dk = 2*delta*(cos(k[1]) - cos(k[2]))   
    res = [ ek  0  0  dk;
            0   ek -dk 0;
            0   -dk -ek 0;
            dk   0   0  -ek
    ]
    return res
end
dimension(hamiltonian::Function) = 4

parameters = Parameters{(:t, :delta)}(0.2,0.0598)
tbafunc = TBA{Fermionic{:BdG}}(lattice, hamiltonian, parameters)
rpa2 = RPA(tbafunc, hilbert, table, (U, ); neighbors=1)
antirpa2 = Algorithm(:dAFM, rpa2);
tespm2 = antirpa2(:chipm, phs);

#plot spectrum of longitudinal spin excitation
plt = plot(
    tespm, 
    xticks=(
        [0, 16/2, 32/2, 48/2, 64/2+1],
        ["(0,0)",  "(π,0)","(π,π)","(0,0)"]
    ),
    clims=(0, 0.5)
)
```