# Surprise.jl - Wrapper to Surprise Python Package

[![][ci-img]][ci-url]
[![][codecov-img]][codecov-url]

**Installation**: at the Julia REPL, `Pkg.add("Surprise")`

**Reporting Issues and Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## Goal

The main aim is to create a framework that facilitates the study of recommender systems in Julia.

## Example

```
julia> using Persa

julia> using DatasetsCF

julia> using Surprise

julia> dataset = DatasetsCF.MovieLens();

julia> model = Surprise.IRSVD(dataset);

julia> Persa.train!(model, dataset)

julia> model[1,1]
Rating: 4.010861934456679 (4)
```

## Algorithms

List of collaborative algorithms:

Algorithm      | Description
-------------|------------------------------------------------------------------------
KNNBasic  | A basic KNN algorithm.
KNNBaseline    | A basic KNN algorithm but using a baseline factor.
KNNWithMeans    | A basic KNN algorithm but using user or item mean.
SlopeOne    | SlopeOne algorithm.
RSVD    | Regulared SVD. The algorithm is also known as SVD by Funk.
IRSVD    | Improved Regulared SVD. Extension of RSVD algorithm adding the user and item bias.

[ci-img]: https://img.shields.io/github/checks-status/JuliaRecsys/Surprise.jl/master?style=flat-square
[ci-url]: https://github.com/JuliaRecsys/Surprise.jl/actions

[codecov-img]: https://img.shields.io/codecov/c/github/JuliaRecsys/Surprise.jl?style=flat-square
[codecov-url]: https://codecov.io/gh/JuliaRecsys/Surprise.jl