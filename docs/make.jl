using RandomPhaseApproximation
using Documenter

DocMeta.setdocmeta!(RandomPhaseApproximation, :DocTestSetup, :(using RandomPhaseApproximation); recursive=true)

makedocs(;
    modules=[RandomPhaseApproximation],
    authors="wwangnju <wwangnju@163.com>",
    repo="https://github.com/Quantum-Many-Body/RandomPhaseApproximation.jl/blob/{commit}{path}#{line}",
    sitename="RandomPhaseApproximation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/RandomPhaseApproximation.jl",
        edit_link="master",
        assets = ["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "examples/Introduction.md",
            "examples/SquarePiFlux.md",
            "examples/Squaredx2y2Wave.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/RandomPhaseApproximation.jl",
    devbranch="master",
)
