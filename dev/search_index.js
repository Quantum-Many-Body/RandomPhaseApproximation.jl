var documenterSearchIndex = {"docs":
[{"location":"examples/Squaredx2y2Wave/","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"CurrentModule = RandomPhaseApproximation","category":"page"},{"location":"examples/Squaredx2y2Wave/#dx2y2-wave-superconductor","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"","category":"section"},{"location":"examples/Squaredx2y2Wave/","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"Spin excitation of t + Δ + Hubbard model on square lattice.","category":"page"},{"location":"examples/Squaredx2y2Wave/#Spectrum-of-spin-excitation","page":"dx2y2-wave superconductor","title":"Spectrum of spin excitation","text":"","category":"section"},{"location":"examples/Squaredx2y2Wave/","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"The following codes could compute the spin excitation within random phase approximation.","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"using QuantumLattices\nusing RandomPhaseApproximation\nusing Plots\nusing TightBindingApproximation: EnergyBands\n\nlattice = Lattice(\n    [0.0, 0.0];\n    vectors=[[1.0, 0.0], [0.0, 1.0]]\n)\nb₁, b₂ = lattice.reciprocals[1], lattice.reciprocals[2]\nhilbert = Hilbert(pid=>Fock{:f}(1, 2) for pid in 1:length(lattice))\n\npair = Coupling(Index(:, FID(:, 1//2, :)), Index(:, FID(:, -1//2, :))) -  Coupling(Index(:, FID(:, -1//2, :)), Index(:, FID(:, 1//2, :)))\n\nt1 = Hopping(:t1, 1.0*0.4, 1)\nΔ = Pairing(\n    :dx2y2, \n    0.299*0.4, \n    1, \n    pair;\n    amplitude=x->(\n        ϕ=azimuth(rcoordinate(x));\n        any(≈(ϕ),(0,π)) && return x[1].site == 1 && x[2].site==2 ? 1 : -1 ;\n        any(≈(ϕ),(π/2, 3π/2)) && return x[1].site == 1 && x[2].site==2 ? -1 : 1 \n    )       \n )\n\nU = Hubbard(:U, 0.4)\nmx = MatrixCoupling(:, FID, :, σ\"x\", :)\nmy = MatrixCoupling(:, FID, :, σ\"y\", :)\nmz = MatrixCoupling(:, FID, :, σ\"z\", :)\n\n#define the RPA frontend \nrpa = RPA(lattice, hilbert, (t1, Δ), (U, ); neighbors=1)\n\n#plot energy bands\npathek = ReciprocalPath{:k}(\n    lattice.reciprocals, \n    (0//2, 0//2)=>(1//2, 0//2), \n    (1//2, 0//2)=>(1//2, 1//2), \n    (1//2, 1//2)=>(0, 0),\n    length=50\n)\netba = Algorithm(:SquareAFM, rpa.tba)(:eband, EnergyBands(pathek, gauge=:rcoordinate))\n\nplt1 = plot(\n    etba, \n    xticks=(\n        [0, 50, 100, 150], \n        [\"(0,0)\", \"(π,0)\",\"(π,π)\",\"(0,0)\"]\n    )\n)\n#display(plt1)","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"The transverse spin excitation is calculated.","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"nx, ny= 16, 16\nbz = ReciprocalZone(lattice.reciprocals, \n    Segment(0, 1, nx), \n    Segment(0//2, 2//2, ny)\n)\npath, = selectpath(\n    [(b₁*0, b₁/2), (b₁/2, b₂/2+b₁/2), (b₂/2+b₁/2, b₂*0)],\n    bz;\n    ends=[false, false, true]\n)\ns⁺ = expand(Onsite(:sx, 1.0+0.0im, 1/2*mx+0.5im*my), bonds(lattice, 1), hilbert, half=false) \ns⁻ = expand(Onsite(:sy, 1.0+0.0im, 1/2*mx-0.5im*my), bonds(lattice, 1), hilbert, half=false) \nsz = expand(Onsite(:sz, 1.0+0.0im, 1/2*mz), bonds(lattice, 1), hilbert, half=false) \n\nphs = ParticleHoleSusceptibility(\n    path, \n    bz, \n    range(0.0, 4.0, length=400), \n    ([s⁺], [s⁻]);\n    η=0.02,\n    save=false,\n    findk=true\n)\nantirpa = Algorithm(:dAFM, rpa);\ntespm = antirpa(:chipm, phs);\n\n#plot spectrum of longitudinal spin excitation\nplt = plot(\n    tespm, \n    xticks=(\n        [0, 16/2, 32/2, 48/2, 64/2+1],\n        [\"(0,0)\",  \"(π,0)\",\"(π,π)\",\"(0,0)\"]\n    ),\n    clims=(0, 0.5)\n)\n#display(plt)","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"Another way to define the TBA.","category":"page"},{"location":"examples/Squaredx2y2Wave/","page":"dx2y2-wave superconductor","title":"dx2y2-wave superconductor","text":"import QuantumLattices: dimension\ntable = Table(hilbert, OperatorUnitToTuple(:site, :orbital, :spin))\nfunction hamiltonian(t::Float64, delta::Float64;k=nothing, kwargs...) \n    @assert !isnothing(k) \"hamiltonian error\"\n    ek = 2*t*(cos(k[1]) + cos(k[2]))\n    dk = 2*delta*(cos(k[1]) - cos(k[2]))   \n    res = [ ek  0  0  dk;\n            0   ek -dk 0;\n            0   -dk -ek 0;\n            dk   0   0  -ek\n    ]\n    return res\nend\ndimension(hamiltonian::Function) = 4\n\nparameters = Parameters{(:t, :delta)}(0.2,0.0598)\ntbafunc = TBA{Fermionic{:BdG}}(lattice, hamiltonian, parameters)\nrpa2 = RPA(tbafunc, hilbert, table, (U, ); neighbors=1)\nantirpa2 = Algorithm(:dAFM, rpa2);\ntespm2 = antirpa2(:chipm, phs);\n\n#plot spectrum of longitudinal spin excitation\nplot(\n    tespm2, \n    xticks=(\n        [0, 16/2, 32/2, 48/2, 64/2+1],\n        [\"(0,0)\",  \"(π,0)\",\"(π,π)\",\"(0,0)\"]\n    ),\n    clims=(0, 0.5)\n)","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"CurrentModule = RandomPhaseApproximation","category":"page"},{"location":"examples/Introduction/#examples","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Here are some examples to illustrate how this package could be used.","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Pages = [\n        \"PiFluxSquare.md\",\n        \"Squaredx2y2Wave.md\"\n        ]\nDepth = 2","category":"page"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"CurrentModule = RandomPhaseApproximation","category":"page"},{"location":"examples/PiFluxSquare/#PiFluxSquare","page":"PiFluxSquare","title":"PiFluxSquare","text":"","category":"section"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"Spin excitation of PiFlux + Bz + Hubbard model on square lattice.","category":"page"},{"location":"examples/PiFluxSquare/#Spectrum-of-spin-excitation","page":"PiFluxSquare","title":"Spectrum of spin excitation","text":"","category":"section"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"The following codes could compute the spin excitation within random phase approximation.","category":"page"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"#using Distributed: @distributed, addprocs, @everywhere\n#addprocs(2)\n#@everywhere \nusing QuantumLattices \nusing RandomPhaseApproximation\nusing Plots\nusing TightBindingApproximation: EnergyBands\n\nlattice = Lattice(\n    [0.0, 0.0], [1.0,0.0];\n    vectors=[[1.0, 1.0], [-1.0, 1.0]]\n)\n\nb₁, b₂ = lattice.reciprocals[1], lattice.reciprocals[2]\nhilbert = Hilbert(pid=>Fock{:f}(1, 2) for pid in 1:length(lattice))\n\n#define pi-flux tight-binding model\nt1 = Hopping(\n    :t1, \n    Complex(0.50), \n    1; \n    amplitude=x->(\n        ϕ = azimuth(rcoordinate(x));\n        any(≈(ϕ), (0,π)) && return x[1].site == 1 && x[2].site==2 ? exp(1im*π/4) : exp(-1im*π/4) ; \n        any(≈(ϕ), (π/2, 3π/2)) && return x[1].site == 1 && x[2].site==2 ? exp(-1im*π/4) : exp(1im*π/4)\n    )\n)\n \nU = Hubbard(:U, 1.6)\nmx = MatrixCoupling(:, FID, :, σ\"x\", :)\nmy = MatrixCoupling(:, FID, :, σ\"y\", :)\nmz = MatrixCoupling(:, FID, :, σ\"z\", :)\nmzz = Onsite(:m0, 0.6, 1/2*mz; amplitude=x->x[1].site==1 ? 1 : -1)\n\n#define the RPA frontend\nrpa = RPA(lattice, hilbert, (t1, mzz), (U, ); neighbors=1)\n\n#define Brillouin zone and the high-symmetry path in the reciprocal zone.\nnx, ny= 12, 12\nbz = ReciprocalZone(lattice.reciprocals, Segment(0, 1, nx), Segment(0//2, 2//2, ny))\npath, = selectpath(\n    [(b₁*0, b₁), (b₁, b₂)],\n    bz;\n    ends=[false,true]\n)\npltpath = plot(bz, path)\n#display(pltpath)","category":"page"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"Calculation of spectra of spin excitation.","category":"page"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"#define the particle-hole channel.\ns⁺ = expand(Onsite(:sx, 1.0+0.0im, 1/2*mx+0.5im*my), bonds(lattice, 2), hilbert, half=false) \ns⁻ = expand(Onsite(:sy, 1.0+0.0im, 1/2*mx-0.5im*my), bonds(lattice, 2), hilbert, half=false) \nsz = expand(Onsite(:sz, 1.0+0.0im, 1/2*mz), bonds(lattice, 2), hilbert, half=false) \n\n#define action of transverse spin-spin susceptibility\nphs = ParticleHoleSusceptibility(\n    path, \n    bz, \n    range(0.0, 4.0, length=200), \n    ([s⁺], [s⁻]); \n    η=0.02,\n    gauge=:rcoordinate, \n    save=false\n)\n\n#define action of longitudinal spin-spin susceptibility\nphsz = ParticleHoleSusceptibility(\n    path, \n    bz, \n    range(0.0, 4.0, length=200), \n    ([sz], [sz]); \n    η=0.02,\n    findk=true\n)\nantirpa = Algorithm(:PiAFM, rpa);\ntespm = antirpa(:chipm, phs);\nteszz = antirpa(:chizz, phsz)\n\n#plot spectrum of spin excitation, i.e. Im[χ⁺⁻(q,ω)]/pi\nplt1 = plot(\n    tespm, \n    xticks=(\n        [0, 6, 12, 18, 25],\n        [\"(0,0)\", \"(π/2,π/2)\", \"(π,π)\",\"(0,π)\",\"(-π,π)\"]\n    ),\n    clims=(0,5)\n)\n#display(plt1)","category":"page"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"The bare spin-spin correlation function is shown as follow.","category":"page"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"#plot Im[χ⁺⁻₀(q, ω)]/pi\nplt2 = plot(\n    tespm, \n    :χ0,\n    xticks=(\n        [0, 6, 12, 18, 25],\n        [\"(0,0)\", \"(π/2,π/2)\", \"(π,π)\",\"(0,π)\",\"(-π,π)\"]\n    ),\n    clims=(0,5)\n)\n#display(plt2)","category":"page"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"#plot Im[χᶻᶻ(q,ω)]/pi\nplt3 = plot(\n    teszz, \n    xticks=(\n        [0, 6, 12, 18, 25],\n        [\"(0,0)\", \"(π/2,π/2)\", \"(π,π)\",\"(0,π)\",\"(-π,π)\"]\n    ),\n    clims=(0,5)\n)\n#display(plt3)","category":"page"},{"location":"examples/PiFluxSquare/#Energy-bands","page":"PiFluxSquare","title":"Energy bands","text":"","category":"section"},{"location":"examples/PiFluxSquare/","page":"PiFluxSquare","title":"PiFluxSquare","text":"#define path in the BZ.\npathek = ReciprocalPath{:k}(\n    lattice.reciprocals, \n    (0//2, 0//2)=>(2//2, 0//2), \n    (2//2, 0//2)=>(0//2, 2//2), \n    length=50\n)\netba = Algorithm(:SquareAFM, rpa.tba )(:band, EnergyBands(pathek))\nplt = plot(\n    etba, \n    xticks=(\n        [0, 25, 50, 75, 100], \n        [\"(0,0)\", \"(π/2,π/2)\", \"(π,π)\",\"(0,π)\",\"(-π,π)\"]\n    )\n)\n#display(plt)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = RandomPhaseApproximation","category":"page"},{"location":"#RandomPhaseApproximation","page":"Home","title":"RandomPhaseApproximation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for RandomPhaseApproximation.","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Standard random phase approximation (particle-hole channel) for quantum lattice systems based on the QuantumLattices and TightBindingApproximation packages.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In Julia v1.8+, please type ] in the REPL to use the package mode, then type this command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/Quantum-Many-Body/RandomPhaseApproximation.jl","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Examples of random phase approximation for quantum lattice system","category":"page"},{"location":"#Manuals","page":"Home","title":"Manuals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [RandomPhaseApproximation]","category":"page"},{"location":"#RandomPhaseApproximation.EigenRPA","page":"Home","title":"RandomPhaseApproximation.EigenRPA","text":"EigenRPA{P<:Union{ReciprocalPath,ReciprocalZone}, RZ<:ReciprocalZone} <: Action\n\nEigenproblem for standard random phase approximation. \n\n\n\n\n\n","category":"type"},{"location":"#RandomPhaseApproximation.EigenRPA-Tuple{Union{QuantumLattices.Spatials.ReciprocalPath, QuantumLattices.Spatials.ReciprocalZone}, QuantumLattices.Spatials.ReciprocalZone}","page":"Home","title":"RandomPhaseApproximation.EigenRPA","text":"EigenRPA(path::Union{ReciprocalPath, ReciprocalZone}, bz::ReciprocalZone; onlyvalue::Bool=true,  options...)\n\nConstruct a EigenRPA type. Attribute options contains (gauge=:icoordinate, exchange=false, η=1e-8, temperature=1e-12, μ=0.0, bnd=nothing)\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.PHVertexRepresentation","page":"Home","title":"RandomPhaseApproximation.PHVertexRepresentation","text":"PHVertexRepresentation{H<:RepresentationGenerator, Vq, Vk, T} <: MatrixRepresentation\n\nMatrix representation of the particle-hole channel of two-body interaction terms. When the k₁ and k₂ are nothing, the exchange terms is ommitted.  1Nsum_kkqalphaeta m nV^ph_alphabetamn(q)-V^ph_alpha mbeta n(k-k)c^dagger_k-qalphac_kbetac^dagger_k nc_k-q m c^†ᵢ = 1/√N∑ₖc†ₖ exp(-ik*rᵢ) \n\n\n\n\n\n","category":"type"},{"location":"#RandomPhaseApproximation.PHVertexRepresentation-Union{Tuple{Any}, Tuple{H}, Tuple{Any, Symbol}} where H<:QuantumLattices.Frameworks.RepresentationGenerator","page":"Home","title":"RandomPhaseApproximation.PHVertexRepresentation","text":"PHVertexRepresentation{H}(table, gauge::Symbol=:icoordinate)  where {H<:RepresentationGenerator}\nPHVertexRepresentation{H}(q, table, gauge::Symbol=:icoordinate) where {H<:RepresentationGenerator}\n\nGet the matrix representation of particle-hole channel.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.ParticleHoleSusceptibility","page":"Home","title":"RandomPhaseApproximation.ParticleHoleSusceptibility","text":"ParticleHoleSusceptibility{P<:Union{ReciprocalPath,ReciprocalZone}, RZ<:ReciprocalZone, E<:AbstractVector, S<:Operators} <: Action\n\nCalculate the particle-hole susceptibility within random phase approximation. Attribute options contains (η=0.01, gauge =:icoordinate, temperature=1e-12, μ=0.0, findk = false)\n\n\n\n\n\n","category":"type"},{"location":"#RandomPhaseApproximation.ParticleHoleSusceptibility-Tuple{Union{QuantumLattices.Spatials.ReciprocalPath, QuantumLattices.Spatials.ReciprocalZone}, QuantumLattices.Spatials.ReciprocalZone, AbstractVector, Tuple{AbstractVector{<:QuantumLattices.QuantumOperators.Operators}, AbstractVector{<:QuantumLattices.QuantumOperators.Operators}}}","page":"Home","title":"RandomPhaseApproximation.ParticleHoleSusceptibility","text":"ParticleHoleSusceptibility(path::Union{ReciprocalPath,ReciprocalZone}, bz::ReciprocalZone, energies::AbstractVector, operators::Tuple{AbstractVector{<:Operators}, AbstractVector{<:Operators}}; options...)\n\nConstruct a ParticleHoleSusceptibility type.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.RPA","page":"Home","title":"RandomPhaseApproximation.RPA","text":"RPA{L<:AbstractTBA, U<:RepresentationGenerator} <: Frontend\n\nRandom phase approximation in a fermionic system.\n\n\n\n\n\n","category":"type"},{"location":"#RandomPhaseApproximation.RPA-Tuple{TightBindingApproximation.AbstractTBA{<:TightBindingApproximation.TBAKind, <:QuantumLattices.Frameworks.OperatorGenerator}, Tuple{Vararg{QuantumLattices.DegreesOfFreedom.Term}}}","page":"Home","title":"RandomPhaseApproximation.RPA","text":"RPA(\n    tba::AbstractTBA{K, <:OperatorGenerator}, \n    uterms::Tuple{Vararg{Term}}\n) where {K<:TBAKind}\nRPA(\n    lattice::AbstractLattice, \n    hilbert::Hilbert, \n    terms::Tuple{Vararg{Term}}, \n    uterms::Tuple{Vararg{Term}}; \n    neighbors::Union{Nothing, Int, Neighbors}=nothing, \n    boundary::Boundary=plain\n)   \nRPA(\n    tba::AbstractTBA{K, <:AnalyticalExpression}, \n    hilbert::Hilbert, \n    table::Table, \n    uterms::Tuple{Vararg{Term}}; \n    neighbors::Union{Nothing, Int, Neighbors}=nothing, \n    boundary::Boundary=plain\n) where {K<:TBAKind}\n\nConstruct a RPA type.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.QuantumOperators.matrix","page":"Home","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(rpa::RPA, field::Symbol=:U; k=nothing, gauge=:icoordinate, kwargs...) -> Matrix\n\nGet matrix of particle-hole channel of interaction.\n\n\n\n\n\n","category":"function"},{"location":"#QuantumLattices.add!-Tuple{AbstractMatrix, PHVertexRepresentation, QuantumLattices.QuantumOperators.Operator}","page":"Home","title":"QuantumLattices.add!","text":"add!(dest::AbstractMatrix, mr::PHVertexRepresentation{<:RepresentationGenerator}, m::Operator; kwargs...)\n\nGet the matrix representation of an operator and add it to destination.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation._chikq0-Tuple{TightBindingApproximation.AbstractTBA, AbstractVector, AbstractVector, Float64}","page":"Home","title":"RandomPhaseApproximation._chikq0","text":"_chikq0(tba::AbstractTBA, k::AbstractVector, q::AbstractVector, omega::Float64; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, kwargs...)\n\nReturn chi0(k,q){ij,mn}= chi0(k,q){-+,+-}, chi0(k,q){12,34}== <c^\\dagger{k,2}c{k-q ,1}c^\\dagger{k-q,3}c_{k,4}> \n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chikqm-Union{Tuple{T}, Tuple{TightBindingApproximation.AbstractTBA, QuantumLattices.Spatials.ReciprocalZone, Union{QuantumLattices.Spatials.ReciprocalPath, QuantumLattices.Spatials.ReciprocalZone}, Array{T, 5}}} where T<:Number","page":"Home","title":"RandomPhaseApproximation.chikqm","text":"chikqm(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalZone, ReciprocalPath}, vph::Array{T,5}; tem::Float64=1e-12, mu::Float64=0.0, onlyvalue::Bool=true, η::Float64=1e-6, bnd::Union{UnitRange{Int}, StepRange{Int,Int}, Vector{Int}, Nothing}=nothing, kwargs...) where T<:Number -> Tuple{ Array{Array{ComplexF64, 1}, 1}, Array{Matrix{ComplexF64}, 1}, Array{Matrix{ComplexF64}, 1}}\n\nGet the eigenvalues, eigenvectors, and unitary transformation (from orbital to band) of particle-hole susceptibilities χ_{αβ}(ω,k1,k2,q). Now only the zero-temperature case is supported.\nvph store the particle-hole vertex,e.g. vph[ndim,ndim,nk,nk,nq] where sqrt(nidm) is the number of degrees of freedom in the unit cell,nk=length(bz), nq=length(path). \ntem is the temperature;\nmu is the chemical potential.\nonlyvalue : only need the eigenvalues ( isval=true,zero temperature ), isval=false denotes that the cholesky method is used.\nη:small number to advoid the semipositive Hamiltonian,i.e. Hamiltonian+diagm([η,η,...])\nbnd:select the bands to calculate the χ₀\nkwargs store the keys which are transfered to matrix function.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chiq-Union{Tuple{T}, Tuple{Array{ComplexF64, 3}, Matrix{Float64}, QuantumLattices.Spatials.ReciprocalZone, Union{QuantumLattices.Spatials.ReciprocalPath, QuantumLattices.Spatials.ReciprocalZone}, Array{T, 3}, AbstractVector}} where T<:Number","page":"Home","title":"RandomPhaseApproximation.chiq","text":"chiq(eigenvc::Array{ComplexF64, 3}, eigenval::Array{Float64, 2}, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, vph::Array{T, 3}, omegam::Vector{Float64}; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false) where T<:Number -> Tuple{ Array{ComplexF64, 4}, Array{ComplexF64, 4} }\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chiq-Union{Tuple{T}, Tuple{Array{T, 3}, Array{ComplexF64, 4}}} where T<:Number","page":"Home","title":"RandomPhaseApproximation.chiq","text":"chiq(vph::Array{T, 3}, chi0::Array{ComplexF64, 4}) where T<:Number -> Array{ComplexF64, 4}\n\nGet the susceptibility χ_{αβ}(ω,q).\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chiq-Union{Tuple{T}, Tuple{TightBindingApproximation.AbstractTBA, QuantumLattices.Spatials.ReciprocalZone, Union{QuantumLattices.Spatials.ReciprocalPath, QuantumLattices.Spatials.ReciprocalZone}, Array{T, 3}, AbstractVector}} where T<:Number","page":"Home","title":"RandomPhaseApproximation.chiq","text":"chiq(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, vph::Array{T,3}, omegam::AbstractVector; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false, kwargs...) where T<:Number -> Tuple{ Array{ComplexF64, 4}, Array{ComplexF64, 4} }\n\nGet the particle-hole susceptibilities χ⁰(ω,q) and χ(ω,q). The spectrum function is satisfied by A(ω,q) = Im[χ(ω+i*0⁺,q)].\n\nArguments\n\nvph is the bare particle-hole vertex (vph[ndim,ndim,nq])\nomegam store the energy points\neta is the magnitude of broaden\ntem is the temperature\nmu is the chemical potential\nscflag == false (default, no superconductivity), or true ( BdG model)\nkwargs is transfered to matrix(tba; kwargs...) function\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.chiq0-Tuple{TightBindingApproximation.AbstractTBA, QuantumLattices.Spatials.ReciprocalZone, Union{QuantumLattices.Spatials.ReciprocalPath, QuantumLattices.Spatials.ReciprocalZone}, AbstractVector}","page":"Home","title":"RandomPhaseApproximation.chiq0","text":"chiq0(tba::AbstractTBA, bz::ReciprocalZone, path::Union{ReciprocalPath, ReciprocalZone}, omegam::Vector{Float64}; eta::Float64=0.01, tem::Float64=1e-12, mu::Float64=0.0, scflag=false, kwargs...) -> Array{Float64, 4}\n\nGet the particle-hole susceptibilities χ⁰(ω,q).\nomegam store the energy points;\neta is the magnitude of broaden;\ntem is the temperature;\nmu is the chemical potential.\n'kwargs' is transfered to matrix(tba;kwargs...) function\nscflag == false(default) => particle-hole channel, ==true => Nambu space.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.correlation-Union{Tuple{S}, Tuple{Array{<:Number, 4}, Union{QuantumLattices.Spatials.ReciprocalPath, QuantumLattices.Spatials.ReciprocalZone}, Tuple{AbstractVector{S}, AbstractVector{S}}, Any}} where S<:(QuantumLattices.QuantumOperators.Operators)","page":"Home","title":"RandomPhaseApproximation.correlation","text":"correlation(χ::Array{<:Number, 4}, path::Union{ReciprocalPath, ReciprocalZone}, operators::Tuple{AbstractVector{S}, AbstractVector{S}}, table; gauge=:rcoordinate) where {S <: Operators} -> Matrix\n\nReturn physical particle-hole susceptibility.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.fermifunc-Union{Tuple{T}, Tuple{T, T}, Tuple{T, T, T}} where T<:Real","page":"Home","title":"RandomPhaseApproximation.fermifunc","text":"fermifunc(e::T, temperature::T=1e-12, mu::T=0.0) where {T<:Real} -> Float64\n\nFermi distribution function.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.findk-Tuple{AbstractVector, QuantumLattices.Spatials.ReciprocalZone}","page":"Home","title":"RandomPhaseApproximation.findk","text":"findk(kq::AbstractVector, bz::ReciprocalZone) -> Int\n\nFind the index of k point in the reduced Brillouin Zone bz, i.e. bz[result] ≈ kq\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.isevenperm-Tuple{Vector}","page":"Home","title":"RandomPhaseApproximation.isevenperm","text":"isevenperm(p::Vector) -> Bool\n\nJudge the number of permutations.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.matrix!-Tuple{Matrix{<:Number}, QuantumLattices.QuantumOperators.Operators, QuantumLattices.DegreesOfFreedom.Table, Any}","page":"Home","title":"RandomPhaseApproximation.matrix!","text":"matrix!(m::Matrix{<:Number}, operators::Operators, table::Table, k; gauge=:rcoordinate)\n\nReturn the matrix representation of operators.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.projchi-Union{Tuple{T1}, Tuple{T}, Tuple{Array{Vector{T}, 1}, Array{Matrix{T1}, 1}, Vector{Matrix{ComplexF64}}, AbstractVector}} where {T<:Number, T1<:Number}","page":"Home","title":"RandomPhaseApproximation.projchi","text":"projchi(val::Array{Array{T, 1}, 1}, vec::Array{Array{T1, 2}, 1}, f2c::Array{ Array{ComplexF64, 2}, 1}, omegam::Vector{Float64}, eta::Float64=1e-2) where {T<:Number,T1<:Number} -> Array{ComplexF64,4}\n\nGet the chi_ijnm(omegaq). The eigenvalues, eigenvectors, and unitary matrix are obtained by methodchikqm`.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.projchiim-Union{Tuple{T1}, Tuple{T}, Tuple{Array{Vector{T}, 1}, Array{Matrix{T1}, 1}, Vector{Matrix{ComplexF64}}, AbstractVector}} where {T<:Number, T1<:Number}","page":"Home","title":"RandomPhaseApproximation.projchiim","text":"projchiim(val::Array{Array{T, 1}, 1}, vec::Array{Array{T1, 2}, 1}, f2c::Array{ Array{ComplexF64, 2}, 1},omegam::Vector{Float64}, eta::Float64=1e-2) where {T<:Number, T1<:Number} -> Array{ComplexF64, 4}\n\nGet the imaginary party of chi_ijnm(omegaq). The eigenvalues,eigenvectors, and unitary matrix are obtained by methodchikqm`.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.selectpath-Tuple{AbstractVector{<:Tuple{var\"#s93\", var\"#s94\"} where {var\"#s93\"<:(AbstractVector), var\"#s94\"<:(AbstractVector)}}, QuantumLattices.Spatials.ReciprocalZone}","page":"Home","title":"RandomPhaseApproximation.selectpath","text":"selectpath(path::AbstractVector{<:Tuple{<:AbstractVector, <:AbstractVector}}, bz::ReciprocalZone;ends::Union{<:AbstractVector{Bool},Nothing}=nothing, atol::Real=atol, rtol::Real=rtol) -> Tuple(Vector{Vector{Float64}}, Vector{Int})  -> -> Tuple(ReciprocalPath, Vector{Int})\n\nSelect a path from the reciprocal zone. Return ReciprocalPath and positions of points in the reciprocal zone.\n\n\n\n\n\n","category":"method"},{"location":"#RandomPhaseApproximation.selectpath-Tuple{Tuple{AbstractVector, AbstractVector}, QuantumLattices.Spatials.ReciprocalZone}","page":"Home","title":"RandomPhaseApproximation.selectpath","text":"selectpath(stsp::Tuple{<:AbstractVector, <:AbstractVector}, rz::ReciprocalZone; ends::Tuple{Bool, Bool}=(true, false), atol::Real=atol, rtol::Real=rtol) -> Tuple(Vector{Vector{Float64}}, Vector{Int})\n\nSelect a path from the reciprocal zone.\n\n\n\n\n\n","category":"method"},{"location":"#RecipesBase.apply_recipe","page":"Home","title":"RecipesBase.apply_recipe","text":"@recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:ParticleHoleSusceptibility}}, mode::Symbol=:χ)\n\nDefine the recipe for the visualization of particle-hole susceptibilities.\n\n\n\n\n\n","category":"function"},{"location":"#RecipesBase.apply_recipe-2","page":"Home","title":"RecipesBase.apply_recipe","text":"@recipe plot(rz::ReciprocalZone, path::Union{ReciprocalPath,Nothing}=nothing)\n\nDefine the recipe for the visualization of a reciprocal zone and a reciprocal path\n\n\n\n\n\n","category":"function"},{"location":"#RecipesBase.apply_recipe-3","page":"Home","title":"RecipesBase.apply_recipe","text":"@recipe plot(pack::Tuple{Algorithm{<:RPA}, Assignment{<:EigenRPA}})\n\nDefine the recipe for the visualization of particle-hole susceptibilities.\n\n\n\n\n\n","category":"function"}]
}
