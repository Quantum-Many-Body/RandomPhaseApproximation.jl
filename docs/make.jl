using RandomPhaseApproximation
using Documenter

DocMeta.setdocmeta!(RandomPhaseApproximation, :DocTestSetup, :(using RandomPhaseApproximation); recursive=true)

makedocs(;
    modules=[RandomPhaseApproximation],
    authors="wwangnju <wwangnju@163.com> and contributors",
    repo="https://github.com/Quantum-Many-Body/RandomPhaseApproximation.jl/blob/{commit}{path}#{line}",
    sitename="RandomPhaseApproximation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/RandomPhaseApproximation.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/RandomPhaseApproximation.jl",
    devbranch="master",
)
