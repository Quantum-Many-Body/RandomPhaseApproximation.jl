```@meta
CurrentModule = RandomPhaseApproximation
```

# ``d_{x^2-y^2}`` wave superconductor on square lattice

Spin excitations of t + Δ + Hubbard model on square lattice by random phase approximation.

## Spectra of spin excitations

The following codes could compute the spin excitation spectra within random phase approximation.

First, construct the RPA frontend:
```@example BdG
using QuantumLattices
using RandomPhaseApproximation
using Plots
using TightBindingApproximation: EnergyBands, Fermionic

lattice = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in 1:length(lattice))
t = Hopping(:t, 1.0*0.4, 1)
Δ = Pairing(
    :Δ,
    0.299*0.4,
    1,
    Coupling(:, FID, :, (1//2, -1//2), :)-Coupling(:, FID, :, (-1//2, 1//2), :);
    amplitude=bond::Bond->(
        ϕ = azimuth(rcoordinate(bond));
        condition = isodd(bond[1].site) && iseven(bond[2].site);
        any(≈(ϕ), (0, π)) && return condition ? 1 : -1;
        any(≈(ϕ), (π/2, 3π/2)) && return condition ? -1 : 1
    )
)
U = Hubbard(:U, 0.4)
rpa = Algorithm(:dx²y², RPA(lattice, hilbert, (t, Δ), (U,); neighbors=1));

nothing # hide
```

The electronic energy bands can be shown as follows:
```@example BdG
# plot electronic energy bands
path = ReciprocalPath(
    reciprocals(lattice), (0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0);
    length=50
)
plot(Algorithm(:dx²y², rpa.frontend.tba)(:EB, EnergyBands(path; gauge=:rcoordinate)))
```

The transverse spin excitation spectra can be computed:
```@example BdG
nk=16
brillouinzone = BrillouinZone(reciprocals(lattice), nk)
path = selectpath(brillouinzone, (0.0, 0.0), (0.0, 0.5), (0.5, 0.5), (0.0, 0.0))[1]

s⁺ = expand(Onsite(:s⁺, 1.0, MatrixCoupling(:, FID, :, σ"+", :)), bonds(lattice, 0), hilbert)
s⁻ = expand(Onsite(:s⁻, 1.0, MatrixCoupling(:, FID, :, σ"-", :)), bonds(lattice, 0), hilbert)

transverse = ParticleHoleSusceptibility(
    path, brillouinzone, range(0.0, 4.0, length=400), ([s⁺], [s⁻]);
    η=0.02, save=false, findk=true
)
plot(rpa(:χ⁺⁻, transverse), clims=(0, 0.5))
```

Another way to define the tight-binding model:
```@example BdG
using TightBindingApproximation: TBA

function hamiltonian(t::Float64, Δ::Float64; k=[0, 0], kwargs...)
    ek = 2t * (cos(k[1])+cos(k[2]))
    dk = 2Δ * (cos(k[1])-cos(k[2]))
    return [ek  0   0   dk;
            0   ek  -dk 0;
            0   -dk -ek 0;
            dk  0   0   -ek
    ]
end

parameters = Parameters{(:t, :Δ)}(0.4, 0.4*0.299)
tba = TBA{Fermionic{:BdG}}(lattice, hamiltonian, parameters)
rpa = Algorithm(:dx²y², RPA(tba, hilbert, U; neighbors=1))
plot(rpa(:χ⁺⁻, transverse), clims=(0, 0.5))
```