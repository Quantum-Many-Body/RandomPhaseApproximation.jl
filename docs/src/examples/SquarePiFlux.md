```@meta
CurrentModule = RandomPhaseApproximation
```

# π-flux state with antiferromagnetic order on square lattice

Spin excitations of π-flux + AFM + Hubbard model on square lattice by random phase approximation.

## Spectra of spin excitations

The following codes could compute the spin excitation spectra within random phase approximation.

First, construct the RPA frontend:
```@example piflux
using QuantumLattices 
using RandomPhaseApproximation
using Plots
using TightBindingApproximation: EnergyBands

lattice = Lattice([0.0, 0.0], [1.0,0.0]; vectors=[[1.0, 1.0], [-1.0, 1.0]])
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in 1:length(lattice))

# define the π-flux tight-binding model and Hubbard interactions
t = Hopping(
    :t,
    Complex(0.50),
    1;
    amplitude=bond::Bond->(
        ϕ = azimuth(rcoordinate(bond));
        condition = isodd(bond[1].site) && iseven(bond[2].site);
        any(≈(ϕ), (0, π)) && return condition ? exp(1im*π/4) : exp(-1im*π/4);
        any(≈(ϕ), (π/2, 3π/2)) && return condition ? exp(-1im*π/4) : exp(1im*π/4)
    )
)
m₀ = Onsite(
    :m₀, 0.6, 1/2*MatrixCoupling(:, FID, :, σ"z", :);
    amplitude=bond::Bond->isodd(bond[1].site) ? 1 : -1
)
U = Hubbard(:U, 1.6)

# define the RPA frontend
rpa = Algorithm(:PiFluxAFM, RPA(lattice, hilbert, (t, m₀), (U,); neighbors=1));

nothing # hide
```

Then, calculate the spin excitation spectra:
```@example piflux
# define the first Brillouin zone and the high-symmetry path in the reciprocal space
nk = 12
brillouinzone = BrillouinZone(reciprocals(lattice), 12)
path, = selectpath(
    brillouinzone, (0, 0)=>(1, 0), (1, 0)=>(0, 1);
    ends=((true, false), (true, true))
)

# define the particle-hole channel operators
s⁺ = expand(Onsite(:s⁺, 1.0, MatrixCoupling(:, FID, :, σ"+", :)), bonds(lattice, 0), hilbert)
s⁻ = expand(Onsite(:s⁻, 1.0, MatrixCoupling(:, FID, :, σ"-", :)), bonds(lattice, 0), hilbert)
sᶻ = expand(Onsite(:sᶻ, 0.5, MatrixCoupling(:, FID, :, σ"z", :)), bonds(lattice, 0), hilbert)

# define and compute the transverse spin-spin susceptibility
χ⁺⁻ = rpa(
    :χ⁺⁻,
    ParticleHoleSusceptibility(
        path,
        brillouinzone,
        range(0.0, 4.0, length=200),
        ([s⁺], [s⁻]);
        η=0.02,
        gauge=:rcoordinate,
        save=false
    )
)

# plot the spectra of transverse spin excitations, i.e. Im[χ⁺⁻(q, ω)]/π
plot(
    χ⁺⁻,
    xticks=(
        [0, 6, 12, 18, 25],
        ["(0, 0)", "(π/2, π/2)", "(π, π)", "(0, π)", "(-π, π)"]
    ),
    clims=(0, 5)
)
```

```@example piflux
# define and compute the longitudinal spin-spin susceptibility
χᶻᶻ = rpa(
    :χᶻᶻ,
    ParticleHoleSusceptibility(
        path,
        brillouinzone,
        range(0.0, 4.0, length=200),
        ([sᶻ], [sᶻ]);
        η=0.02,
        findk=true
    )
)

# plot the spectra of longitudinal spin excitations, i.e. Im[χᶻᶻ(q, ω)]/π
plot(
    χᶻᶻ;
    xticks=(
        [0, 6, 12, 18, 25],
        ["(0, 0)", "(π/2, π/2)", "(π, π)", "(0, π)", "(-π, π)"]
    ),
    clims=(0, 5)
)
```

The bare spin-spin correlation functions are shown as follows:
```@example piflux
# plot Im[χ⁺⁻₀(q, ω)]/π
plot(
    χ⁺⁻,
    :χ0;
    xticks=(
        [0, 6, 12, 18, 25],
        ["(0, 0)", "(π/2, π/2)", "(π, π)", "(0, π)", "(-π, π)"]
    ),
    clims=(0, 5)
)
```

```@example piflux
# plot Im[χᶻᶻ₀(q, ω)]/π
plot(
    χᶻᶻ,
    :χ0;
    xticks=(
        [0, 6, 12, 18, 25],
        ["(0, 0)", "(π/2, π/2)", "(π, π)", "(0, π)", "(-π, π)"]
    ),
    clims=(0, 5)
)
```

## Electronic energy bands

The electronic energy bands of the model:

```@example piflux
# define a path in the Brillouin zone
path = ReciprocalPath(
    reciprocals(lattice),
    (0//2, 0//2)=>(2//2, 0//2),
    (2//2, 0//2)=>(0//2, 2//2);
    length=50
)
ebs = Algorithm(:PiFluxAFM, rpa.frontend.tba)(:EB, EnergyBands(path))
plot(
    ebs;
    xticks=(
        [0, 25, 50, 75, 100],
        ["(0, 0)", "(π/2, π/2)", "(π, π)", "(0, π)", "(-π, π)"]
    )
)
```
