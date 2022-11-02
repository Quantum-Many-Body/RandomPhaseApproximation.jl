```@meta
CurrentModule = RandomPhaseApproximation
```

# PiFluxSquare
Spin excitation of PiFlux + Bz + Hubbard model on square lattice.

## Spectrum of spin excitation

The following codes could compute the spin excitation within random phase approximation.

```@example piflux
#using Distributed: @distributed, addprocs, @everywhere
#addprocs(2)
#@everywhere 
using QuantumLattices 
using RandomPhaseApproximation
using Plots
using TightBindingApproximation: EnergyBands

lattice = Lattice(
    [0.0, 0.0], [1.0,0.0];
    vectors=[[1.0, 1.0], [-1.0, 1.0]]
)

b₁, b₂ = lattice.reciprocals[1], lattice.reciprocals[2]
hilbert = Hilbert(pid=>Fock{:f}(1, 2) for pid in 1:length(lattice))

#define pi-flux tight-binding model
t1 = Hopping(
    :t1, 
    Complex(0.50), 
    1; 
    amplitude=x->(
        ϕ = azimuth(rcoordinate(x));
        any(≈(ϕ), (0,π)) && return x[1].site == 1 && x[2].site==2 ? exp(1im*π/4) : exp(-1im*π/4) ; 
        any(≈(ϕ), (π/2, 3π/2)) && return x[1].site == 1 && x[2].site==2 ? exp(-1im*π/4) : exp(1im*π/4)
    )
)
 
U = Hubbard(:U, 1.6)
mx = MatrixCoupling(:, FID, :, σ"x", :)
my = MatrixCoupling(:, FID, :, σ"y", :)
mz = MatrixCoupling(:, FID, :, σ"z", :)
mzz = Onsite(:m0, 0.6, 1/2*mz; amplitude=x->x[1].site==1 ? 1 : -1)

#define the RPA frontend
rpa = RPA(lattice, hilbert, (t1, mzz), (U, ); neighbors=1)

#define Brillouin zone and the high-symmetry path in the reciprocal zone.
nx, ny= 12, 12
bz = ReciprocalZone(lattice.reciprocals, Segment(0, 1, nx), Segment(0//2, 2//2, ny))
path, = selectpath(
    [(b₁*0, b₁), (b₁, b₂)],
    bz;
    ends=[false,true]
)
pltpath = plot(bz, path)
#display(pltpath)
```

Calculation of spectra of spin excitation.
```@example piflux
#define the particle-hole channel.
s⁺ = expand(Onsite(:sx, 1.0+0.0im, 1/2*mx+0.5im*my), bonds(lattice, 2), hilbert, half=false) 
s⁻ = expand(Onsite(:sy, 1.0+0.0im, 1/2*mx-0.5im*my), bonds(lattice, 2), hilbert, half=false) 
sz = expand(Onsite(:sz, 1.0+0.0im, 1/2*mz), bonds(lattice, 2), hilbert, half=false) 

#define action of transverse spin-spin susceptibility
phs = ParticleHoleSusceptibility(
    path, 
    bz, 
    range(0.0, 4.0, length=200), 
    ([s⁺], [s⁻]); 
    η=0.02,
    gauge=:rcoordinate, 
    save=false
)

#define action of longitudinal spin-spin susceptibility
phsz = ParticleHoleSusceptibility(
    path, 
    bz, 
    range(0.0, 4.0, length=200), 
    ([sz], [sz]); 
    η=0.02,
    findk=true
)
antirpa = Algorithm(:PiAFM, rpa);
tespm = antirpa(:chipm, phs);
teszz = antirpa(:chizz, phsz)

#plot spectrum of spin excitation, i.e. Im[χ⁺⁻(q,ω)]/pi
plt1 = plot(
    tespm, 
    xticks=(
        [0, 6, 12, 18, 25],
        ["(0,0)", "(π/2,π/2)", "(π,π)","(0,π)","(-π,π)"]
    ),
    clims=(0,5)
)
#display(plt1)
```
The bare spin-spin correlation function is shown as follow.
```@example piflux
#plot Im[χ⁺⁻₀(q, ω)]/pi
plt2 = plot(
    tespm, 
    :χ0,
    xticks=(
        [0, 6, 12, 18, 25],
        ["(0,0)", "(π/2,π/2)", "(π,π)","(0,π)","(-π,π)"]
    ),
    clims=(0,5)
)
#display(plt2)
```

```@example piflux
#plot Im[χᶻᶻ(q,ω)]/pi
plt3 = plot(
    teszz, 
    xticks=(
        [0, 6, 12, 18, 25],
        ["(0,0)", "(π/2,π/2)", "(π,π)","(0,π)","(-π,π)"]
    ),
    clims=(0,5)
)
#display(plt3)
```
## Energy bands
```@example piflux
#define path in the BZ.
pathek = ReciprocalPath{:k}(
    lattice.reciprocals, 
    (0//2, 0//2)=>(2//2, 0//2), 
    (2//2, 0//2)=>(0//2, 2//2), 
    length=50
)
etba = Algorithm(:SquareAFM, rpa.tba )(:band, EnergyBands(pathek))
plt = plot(
    etba, 
    xticks=(
        [0, 25, 50, 75, 100], 
        ["(0,0)", "(π/2,π/2)", "(π,π)","(0,π)","(-π,π)"]
    )
)
#display(plt)
```
