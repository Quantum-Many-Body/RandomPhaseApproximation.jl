var documenterSearchIndex = {"docs":
[{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"CurrentModule = RandomPhaseApproximation","category":"page"},{"location":"examples/Squaredx2y2Wave/#d_{x2-y2}-wave-superconductor-on-square-lattice","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"","category":"section"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"Spin excitations of t + Δ + Hubbard model on square lattice by random phase approximation.","category":"page"},{"location":"examples/Squaredx2y2Wave/#Spectra-of-spin-excitations","page":"d_x^2-y^2 wave superconductor on square lattice","title":"Spectra of spin excitations","text":"","category":"section"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"The following codes could compute the spin excitation spectra within random phase approximation.","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"First, construct the RPA frontend:","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"using QuantumLattices\nusing RandomPhaseApproximation\nusing Plots\nusing TightBindingApproximation: EnergyBands, Fermionic\n\nlattice = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])\nhilbert = Hilbert(site=>Fock{:f}(1, 2) for site in 1:length(lattice))\nt = Hopping(:t, 1.0*0.4, 1)\nΔ = Pairing(\n    :Δ,\n    0.299*0.4,\n    1,\n    Coupling(:, FID, :, (1//2, -1//2), :)-Coupling(:, FID, :, (-1//2, 1//2), :);\n    amplitude=bond::Bond->(\n        ϕ = azimuth(rcoordinate(bond));\n        condition = isodd(bond[1].site) && iseven(bond[2].site);\n        any(≈(ϕ), (0, π)) && return condition ? 1 : -1;\n        any(≈(ϕ), (π/2, 3π/2)) && return condition ? -1 : 1\n    )\n)\nU = Hubbard(:U, 0.4)\nrpa = Algorithm(:dx²y², RPA(lattice, hilbert, (t, Δ), (U,); neighbors=1));\n\nnothing # hide","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"The electronic energy bands can be shown as follows:","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"# plot electronic energy bands\npath = ReciprocalPath(\n    reciprocals(lattice),\n    (0//2, 0//2)=>(1//2, 0//2),\n    (1//2, 0//2)=>(1//2, 1//2),\n    (1//2, 1//2)=>(0, 0);\n    length=50\n)\nebs = Algorithm(:dx²y², rpa.frontend.tba)(:EB, EnergyBands(path, gauge=:rcoordinate))\nplot(\n    ebs,\n    xticks=(\n        [0, 50, 100, 150],\n        [\"(0, 0)\", \"(π, 0)\", \"(π, π)\", \"(0, 0)\"]\n    )\n)","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"The transverse spin excitation spectra can be computed:","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"nk=16\nbrillouinzone = BrillouinZone(reciprocals(lattice), nk)\npath, = selectpath(\n    brillouinzone,\n    (0, 0)=>(0, 1//2),\n    (0, 1//2)=>(1//2, 1//2),\n    (1//2, 1//2)=>(0, 0);\n    ends=((true, false), (true, false), (true, true))\n)\n\ns⁺ = expand(Onsite(:s⁺, 1.0, MatrixCoupling(:, FID, :, σ\"+\", :)), bonds(lattice, 0), hilbert)\ns⁻ = expand(Onsite(:s⁻, 1.0, MatrixCoupling(:, FID, :, σ\"-\", :)), bonds(lattice, 0), hilbert)\n\ntransverse = ParticleHoleSusceptibility(\n    path,\n    brillouinzone,\n    range(0.0, 4.0, length=400),\n    ([s⁺], [s⁻]);\n    η=0.02,\n    save=false,\n    findk=true\n)\nχ⁺⁻ = rpa(:χ⁺⁻, transverse)\n\nplot(\n    χ⁺⁻,\n    xticks=(\n        [0, 16/2, 32/2, 48/2, 64/2+1],\n        [\"(0, 0)\", \"(π, 0)\", \"(π, π)\", \"(0, 0)\"]\n    ),\n    clims=(0, 0.5)\n)","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"Another way to define the tight-binding model:","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"d_x^2-y^2 wave superconductor on square lattice","title":"d_x^2-y^2 wave superconductor on square lattice","text":"import QuantumLattices: dimension\nusing TightBindingApproximation: TBA\n\nfunction hamiltonian(t::Float64, Δ::Float64; k=nothing, kwargs...) \n    @assert !isnothing(k) \"hamiltonian error\"\n    ek = 2t * (cos(k[1])+cos(k[2]))\n    dk = 2Δ * (cos(k[1])-cos(k[2]))\n    return [ek  0   0   dk;\n            0   ek  -dk 0;\n            0   -dk -ek 0;\n            dk  0   0   -ek\n    ]\nend\n@inline dimension(::typeof(hamiltonian)) = 4\n\nparameters = Parameters{(:t, :Δ)}(0.4, 0.4*0.299)\ntba = TBA{Fermionic{:BdG}}(lattice, hamiltonian, parameters)\nrpa = Algorithm(:dx²y², RPA(tba, hilbert, (U,); neighbors=1))\nχ⁺⁻ = rpa(:χ⁺⁻, transverse)\nplot(\n    χ⁺⁻,\n    xticks=(\n        [0, 16/2, 32/2, 48/2, 64/2+1],\n        [\"(0, 0)\", \"(π, 0)\", \"(π, π)\", \"(0, 0)\"]\n    ),\n    clims=(0, 0.5)\n)","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"CurrentModule = RandomPhaseApproximation","category":"page"},{"location":"examples/Introduction/#examples","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Here are some examples to illustrate how this package could be used.","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Pages = [\n        \"SquarePiFlux.md\",\n        \"Squaredx2y2Wave.md\"\n        ]\nDepth = 2","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"CurrentModule = RandomPhaseApproximation","category":"page"},{"location":"examples/SquarePiFlux/#π-flux-state-with-antiferromagnetic-order-on-square-lattice","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"","category":"section"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"Spin excitations of π-flux + AFM + Hubbard model on square lattice by random phase approximation.","category":"page"},{"location":"examples/SquarePiFlux/#Spectra-of-spin-excitations","page":"π-flux state with antiferromagnetic order on square lattice","title":"Spectra of spin excitations","text":"","category":"section"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"The following codes could compute the spin excitation spectra within random phase approximation.","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"First, construct the RPA frontend:","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"using QuantumLattices \nusing RandomPhaseApproximation\nusing Plots\nusing TightBindingApproximation: EnergyBands\n\nlattice = Lattice([0.0, 0.0], [1.0,0.0]; vectors=[[1.0, 1.0], [-1.0, 1.0]])\nhilbert = Hilbert(site=>Fock{:f}(1, 2) for site in 1:length(lattice))\n\n# define the π-flux tight-binding model and Hubbard interactions\nt = Hopping(\n    :t,\n    Complex(0.50),\n    1;\n    amplitude=bond::Bond->(\n        ϕ = azimuth(rcoordinate(bond));\n        condition = isodd(bond[1].site) && iseven(bond[2].site);\n        any(≈(ϕ), (0, π)) && return condition ? exp(1im*π/4) : exp(-1im*π/4);\n        any(≈(ϕ), (π/2, 3π/2)) && return condition ? exp(-1im*π/4) : exp(1im*π/4)\n    )\n)\nm₀ = Onsite(\n    :m₀, 0.6, 1/2*MatrixCoupling(:, FID, :, σ\"z\", :);\n    amplitude=bond::Bond->isodd(bond[1].site) ? 1 : -1\n)\nU = Hubbard(:U, 1.6)\n\n# define the RPA frontend\nrpa = Algorithm(:PiFluxAFM, RPA(lattice, hilbert, (t, m₀), (U,); neighbors=1));\n\nnothing # hide","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"Then, calculate the spin excitation spectra:","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"# define the first Brillouin zone and the high-symmetry path in the reciprocal space\nnk = 12\nbrillouinzone = BrillouinZone(reciprocals(lattice), 12)\npath, = selectpath(\n    brillouinzone, (0, 0)=>(1, 0), (1, 0)=>(0, 1);\n    ends=((true, false), (true, true))\n)\n\n# define the particle-hole channel operators\ns⁺ = expand(Onsite(:s⁺, 1.0, MatrixCoupling(:, FID, :, σ\"+\", :)), bonds(lattice, 0), hilbert)\ns⁻ = expand(Onsite(:s⁻, 1.0, MatrixCoupling(:, FID, :, σ\"-\", :)), bonds(lattice, 0), hilbert)\nsᶻ = expand(Onsite(:sᶻ, 0.5, MatrixCoupling(:, FID, :, σ\"z\", :)), bonds(lattice, 0), hilbert)\n\n# define and compute the transverse spin-spin susceptibility\nχ⁺⁻ = rpa(\n    :χ⁺⁻,\n    ParticleHoleSusceptibility(\n        path,\n        brillouinzone,\n        range(0.0, 4.0, length=200),\n        ([s⁺], [s⁻]);\n        η=0.02,\n        gauge=:rcoordinate,\n        save=false\n    )\n)\n\n# plot the spectra of transverse spin excitations, i.e. Im[χ⁺⁻(q, ω)]/π\nplot(\n    χ⁺⁻,\n    xticks=(\n        [0, 6, 12, 18, 25],\n        [\"(0, 0)\", \"(π/2, π/2)\", \"(π, π)\", \"(0, π)\", \"(-π, π)\"]\n    ),\n    clims=(0, 5)\n)","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"# define and compute the longitudinal spin-spin susceptibility\nχᶻᶻ = rpa(\n    :χᶻᶻ,\n    ParticleHoleSusceptibility(\n        path,\n        brillouinzone,\n        range(0.0, 4.0, length=200),\n        ([sᶻ], [sᶻ]);\n        η=0.02,\n        findk=true\n    )\n)\n\n# plot the spectra of longitudinal spin excitations, i.e. Im[χᶻᶻ(q, ω)]/π\nplot(\n    χᶻᶻ;\n    xticks=(\n        [0, 6, 12, 18, 25],\n        [\"(0, 0)\", \"(π/2, π/2)\", \"(π, π)\", \"(0, π)\", \"(-π, π)\"]\n    ),\n    clims=(0, 5)\n)","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"The bare spin-spin correlation functions are shown as follows:","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"# plot Im[χ⁺⁻₀(q, ω)]/π\nplot(\n    χ⁺⁻,\n    :χ0;\n    xticks=(\n        [0, 6, 12, 18, 25],\n        [\"(0, 0)\", \"(π/2, π/2)\", \"(π, π)\", \"(0, π)\", \"(-π, π)\"]\n    ),\n    clims=(0, 5)\n)","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"# plot Im[χᶻᶻ₀(q, ω)]/π\nplot(\n    χᶻᶻ,\n    :χ0;\n    xticks=(\n        [0, 6, 12, 18, 25],\n        [\"(0, 0)\", \"(π/2, π/2)\", \"(π, π)\", \"(0, π)\", \"(-π, π)\"]\n    ),\n    clims=(0, 5)\n)","category":"page"},{"location":"examples/SquarePiFlux/#Electronic-energy-bands","page":"π-flux state with antiferromagnetic order on square lattice","title":"Electronic energy bands","text":"","category":"section"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"The electronic energy bands of the model:","category":"page"},{"location":"examples/SquarePiFlux/","page":"π-flux state with antiferromagnetic order on square lattice","title":"π-flux state with antiferromagnetic order on square lattice","text":"# define a path in the Brillouin zone\npath = ReciprocalPath(\n    reciprocals(lattice),\n    (0//2, 0//2)=>(2//2, 0//2),\n    (2//2, 0//2)=>(0//2, 2//2);\n    length=50\n)\nebs = Algorithm(:PiFluxAFM, rpa.frontend.tba)(:EB, EnergyBands(path))\nplot(\n    ebs;\n    xticks=(\n        [0, 25, 50, 75, 100],\n        [\"(0, 0)\", \"(π/2, π/2)\", \"(π, π)\", \"(0, π)\", \"(-π, π)\"]\n    )\n)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = RandomPhaseApproximation","category":"page"},{"location":"#RandomPhaseApproximation","page":"Home","title":"RandomPhaseApproximation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for RandomPhaseApproximation.","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Standard random phase approximation (particle-hole channel) for quantum lattice systems based on the QuantumLattices and TightBindingApproximation packages.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In Julia v1.8+, please type ] in the REPL to use the package mode, then type this command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/Quantum-Many-Body/RandomPhaseApproximation.jl","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Examples of random phase approximation for quantum lattice system","category":"page"},{"location":"#Manuals","page":"Home","title":"Manuals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [RandomPhaseApproximation]\nOrder = [:module, :constant, :type, :macro, :function]","category":"page"},{"location":"#RandomPhaseApproximation.EigenRPA","page":"Home","title":"RandomPhaseApproximation.EigenRPA","text":"EigenRPA{P<:ReciprocalSpace, B<:BrillouinZone} <: Action\n\nEigen problem for standard random phase approximation.\n\n\n\n\n\n","category":"type"},{"location":"#RandomPhaseApproximation.EigenRPA-Tuple{QuantumLattices.Spatials.ReciprocalSpace, QuantumLattices.Spatials.BrillouinZone}","page":"Home","title":"RandomPhaseApproximation.EigenRPA","text":"EigenRPA(reciprocalspace::ReciprocalSpace, brillouinzone::BrillouinZone; eigvals_only::Bool=true, options...)\n\nConstruct a EigenRPA type. Attribute options contains (gauge=:icoordinate, exchange=false, η=1e-8, temperature=1e-12, μ=0.0, bands=nothing).\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.PHVertexMatrix","page":"Home","title":"RandomPhaseApproximation.PHVertexMatrix","text":"PHVertexMatrix{D<:Number, Vq, Vk, T} <: MatrixRepresentation\n\nMatrix representation of the particle-hole channel of two-body interaction terms:\n\nfrac1Nsum_k₁k₂q  alphabeta m nV^ph_alphabeta  mn(q)-V^ph_alpha m  beta n(k₂-k₁)c^dagger_k₁-q  alphac_k₁  betac^dagger_k₂  nc_k₂-q  m\n\nWhen the k₁ and k₂ are nothing, the exchange terms are omitted. Here, the Fourier transformation reads:\n\nc^_i = frac1sqrtN _k c^_k exp(-i k rᵢ)\n\n\n\n\n\n","category":"type"},{"location":"#RandomPhaseApproximation.PHVertexMatrix-Union{Tuple{Any}, Tuple{D}, Tuple{Any, Symbol}} where D<:Number","page":"Home","title":"RandomPhaseApproximation.PHVertexMatrix","text":"PHVertexMatrix{D}(q, k₁, k₂, table, gauge::Symbol=:icoordinate) where {D<:Number}\nPHVertexMatrix{D}(table, gauge::Symbol=:icoordinate)  where {D<:Number}\nPHVertexMatrix{D}(q, table, gauge::Symbol=:icoordinate) where {D<:Number}\n\nGet the matrix representation of particle-hole channel.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.ParticleHoleSusceptibility","page":"Home","title":"RandomPhaseApproximation.ParticleHoleSusceptibility","text":"ParticleHoleSusceptibility{P<:ReciprocalSpace, B<:BrillouinZone, E<:AbstractVector, S<:Operators} <: Action\n\nCalculate the particle-hole susceptibility within random phase approximation.\n\nAttribute options contains (η=0.01, gauge=:icoordinate, temperature=1e-12, μ=0.0, findk=false).\n\n\n\n\n\n","category":"type"},{"location":"#RandomPhaseApproximation.ParticleHoleSusceptibility-Tuple{QuantumLattices.Spatials.ReciprocalSpace, QuantumLattices.Spatials.BrillouinZone, AbstractVector, Tuple{Vector{<:QuantumLattices.QuantumOperators.Operators}, Vector{<:QuantumLattices.QuantumOperators.Operators}}}","page":"Home","title":"RandomPhaseApproximation.ParticleHoleSusceptibility","text":"ParticleHoleSusceptibility(reciprocalspace::ReciprocalSpace, brillouinzone::BrillouinZone, energies::Vector, operators::Tuple{Vector{<:Operators}, Vector{<:Operators}}; options...)\n\nConstruct a ParticleHoleSusceptibility type.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.RPA","page":"Home","title":"RandomPhaseApproximation.RPA","text":"RPA{L<:AbstractTBA, U<:RepresentationGenerator} <: Frontend\n\nRandom phase approximation in a fermionic system.\n\n\n\n\n\n","category":"type"},{"location":"#RandomPhaseApproximation.RPA-Tuple{TightBindingApproximation.AbstractTBA, Tuple{Vararg{QuantumLattices.DegreesOfFreedom.Term}}}","page":"Home","title":"RandomPhaseApproximation.RPA","text":"RPA(tba::AbstractTBA, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing)\nRPA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing)\nRPA(tba::AbstractTBA{K, <:AnalyticalExpression}, hilbert::Hilbert, interactions::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing) where {K<:TBAKind}\n\nConstruct an RPA type.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.QuantumOperators.matrix","page":"Home","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(rpa::RPA, field::Symbol=:int; k=nothing, gauge=:icoordinate, kwargs...) -> Matrix\n\nGet matrix of particle-hole channel of interaction.\n\n\n\n\n\n","category":"function"},{"location":"#QuantumLattices.add!-Tuple{AbstractMatrix, PHVertexMatrix, QuantumLattices.QuantumOperators.Operator}","page":"Home","title":"QuantumLattices.add!","text":"add!(dest::AbstractMatrix, mr::PHVertexMatrix, m::Operator; kwargs...)\n\nGet the matrix representation of an operator and add it to destination.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chiq-Tuple{Array{<:Number, 3}, Array{ComplexF64, 4}}","page":"Home","title":"RandomPhaseApproximation.chiq","text":"chiq(vph::Array{<:Number, 3}, chi0::Array{ComplexF64, 4}) -> Array{ComplexF64, 4}\n\nGet the susceptibility χ_αβ(ω q).\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chiq0-Tuple{TightBindingApproximation.AbstractTBA, QuantumLattices.Spatials.BrillouinZone, QuantumLattices.Spatials.ReciprocalSpace, AbstractVector}","page":"Home","title":"RandomPhaseApproximation.chiq0","text":"chiq0(tba::AbstractTBA, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, energies::Vector{Float64}; η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false, kwargs...) -> Array{Float64, 4}\n\nGet the particle-hole susceptibilities χ⁰(ω, q).\n\nArguments\n\nenergies: the energy points\nη: the magnitude of broaden\ntemperature: the temperature\nμ: the chemical potential\nscflag: false(default) for particle-hole channel, true for Nambu space\nkwargs: the keyword arguments transferred to the matrix(tba; kwargs...) function\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chiq0chiq-Tuple{Array{ComplexF64, 3}, Matrix{Float64}, QuantumLattices.Spatials.BrillouinZone, QuantumLattices.Spatials.ReciprocalSpace, Array{<:Number, 3}, AbstractVector}","page":"Home","title":"RandomPhaseApproximation.chiq0chiq","text":"chiq0chiq(\n    eigvecs::Array{ComplexF64, 3}, eigvals::Array{Float64, 2}, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 3}, energies::Vector{Float64};\n    η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false\n) -> Tuple{Array{ComplexF64, 4}, Array{ComplexF64, 4}}\n\nGet the particle-hole susceptibilities χ⁰(ω, q) and χ(ω, q). The spectral function is satisfied by A(ω q) = textImχ(ω+i0 q).\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chiq0chiq-Tuple{TightBindingApproximation.AbstractTBA, AbstractVector{<:AbstractVector}, QuantumLattices.Spatials.ReciprocalSpace, Array{<:Number, 3}, AbstractVector}","page":"Home","title":"RandomPhaseApproximation.chiq0chiq","text":"chiq0chiq(\n    tba::AbstractTBA, brillouinzone::AbstractVector{<:AbstractVector}, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 3}, energies::AbstractVector;\n    η::Float64=0.01, temperature::Float64=1e-12, μ::Float64=0.0, scflag=false, kwargs...\n) -> Tuple{Array{ComplexF64, 4}, Array{ComplexF64, 4}}\n\nGet the particle-hole susceptibilities χ⁰(ω, q) and χ(ω, q). The spectral function is satisfied by A(ω q) = textImχ(ω+i0 q).\n\nArguments\n\nvph: the bare particle-hole vertex (vph[ndim, ndim, nq])\nenergies: the energy points\nη: the magnitude of broaden\ntemperature: the temperature\nμ: the chemical potential\nscflag: false (default, no superconductivity), or true (BdG model)\nkwargs: the keyword arguments transferred to the matrix(tba; kwargs...) function\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.correlation-Tuple{Array{<:Number, 4}, QuantumLattices.Spatials.ReciprocalSpace, Tuple{Vector{<:QuantumLattices.QuantumOperators.Operators}, Vector{<:QuantumLattices.QuantumOperators.Operators}}, Any}","page":"Home","title":"RandomPhaseApproximation.correlation","text":"correlation(χ::Array{<:Number, 4}, reciprocalspace::ReciprocalSpace, operators::Tuple{Vector{<:Operators}, Vector{<:Operators}}, table; gauge=:rcoordinate) -> Matrix\n\nReturn physical particle-hole susceptibility.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.eigenrpa-Tuple{TightBindingApproximation.AbstractTBA, QuantumLattices.Spatials.BrillouinZone, QuantumLattices.Spatials.ReciprocalSpace, Array{<:Number, 5}}","page":"Home","title":"RandomPhaseApproximation.eigenrpa","text":"eigenrpa(\n    tba::AbstractTBA, brillouinzone::BrillouinZone, reciprocalspace::ReciprocalSpace, vph::Array{<:Number, 5};\n    temperature::Float64=1e-12, μ::Float64=0.0, eigvals_only::Bool=true, η::Float64=1e-6, bands::Union{UnitRange{Int}, StepRange{Int,Int}, Vector{Int}, Nothing}=nothing, kwargs...\n) -> Tuple{Array{Array{ComplexF64, 1}, 1}, Array{Matrix{ComplexF64}, 1}, Array{Matrix{ComplexF64}, 1}}\n\nGet the eigenvalues, eigenvectors, and unitary transformation (from orbital to band) of particle-hole susceptibilities χ_αβ(ω k₁ k₂ q).\n\nNow only the zero-temperature case is supported.\n\nArguments\n\nvph: the particle-hole vertex, e.g. vph[ndim,ndim,nk,nk,nq] where sqrt(ndim) is the number of degrees of freedom in the unit cell, nk=length(brillouinzone), nq=length(reciprocalspace)\ntemperature: the temperature\nμ: the chemical potential\neigvals_only: only the eigenvalues needed, when it is false the cholesky method is used\nη: the small number to avoid the semi-positive Hamiltonian, i.e. Hamiltonian+diagm([η, η, ...])\nbands: the selected bands to calculate the χ₀\nkwargs: the keyword arguments transferred to the matrix(tba; kwargs...) function\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.fermifunc","page":"Home","title":"RandomPhaseApproximation.fermifunc","text":"fermifunc(e::Real, temperature::Real=1e-12, μ::Real=0.0) -> Float64\n\nFermi distribution function. Boltzmann constant k_B=1.\n\n\n\n\n\n","category":"function"},{"location":"#RandomPhaseApproximation.isevenperm-Tuple{Vector}","page":"Home","title":"RandomPhaseApproximation.isevenperm","text":"isevenperm(p::Vector) -> Bool\n\nJudge the parity of permutations.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.issamesite-Tuple{QuantumLattices.DegreesOfFreedom.CompositeIndex, QuantumLattices.DegreesOfFreedom.CompositeIndex}","page":"Home","title":"RandomPhaseApproximation.issamesite","text":"issamesite(op₁::CompositeIndex, op₂::CompositeIndex) -> Bool\n\nJudge whether two composite indices are on site same site.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.vertex","page":"Home","title":"RandomPhaseApproximation.vertex","text":"vertex(rpa::RPA, reciprocalspace::AbstractVector{<:AbstractVector}, gauge=:icoordinate) -> Array{<:Number, 3}\n\nReturn particle-hole vertex induced by the direct channel of interaction (except the Hubbard interaction which include both direct and exchange channels).\n\n\n\n\n\n","category":"function"},{"location":"#RecipesBase.apply_recipe","page":"Home","title":"RecipesBase.apply_recipe","text":"@recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}}, mode::Symbol=:χ)\n\nDefine the recipe for the visualization of particle-hole susceptibilities.\n\n\n\n\n\n","category":"function"},{"location":"#RecipesBase.apply_recipe-2","page":"Home","title":"RecipesBase.apply_recipe","text":"@recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}}, ecut::Float64, dE::Float64=1e-3, mode::Symbol=:χ, reim::Symbol=:re)\n\n\n\n\n\n","category":"function"},{"location":"#RecipesBase.apply_recipe-3","page":"Home","title":"RecipesBase.apply_recipe","text":"@recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:EigenRPA}})\n\nDefine the recipe for the visualization of particle-hole susceptibilities.\n\n\n\n\n\n","category":"function"}]
}
