
#TODO: the inverse transform logic need be prepared
struct pipeline
    transforms
end

Pipeline(Fn) = pipeline(Tuple([Fn]))

function Pipeline( FnStack... )
    pipe = Array{Any,1}( undef, length( FnStack ))
    for (i, fn) in enumerate( FnStack )
        pipe[ i ] = fn
    end
    return pipeline(Tuple( pipe ))
end

function (P::pipeline)(X)
	foldl((X, p) -> p(X), P.transforms, init = X)
end

struct Dataset
    data::AbstractVector
    tfms::NTuple
end

#TODO: Code can be better at checking the
#no. of pipelines
function Dataset(data; tfms=nothing)
    if length(tfms) == 2
        Dataset(data, (Pipeline(tfms[1]...), Pipeline(tfms[2]...)))
    else
        Dataset(data, (Pipeline(tfms[1]...)))
    end 
end

nobs(dataset::Dataset) = length(dataset.data)

#TODO:Replace indexing with first and last
function getobs(dataset::Dataset, idx::Int64)
    data = dataset.data[idx]

    if dataset.tfms[1]==nothing || dataset.tfms[2]==nothing
       (dataset.tfms[1](data))
    else
       (dataset.tfms[1](data), dataset.tfms[2](data))
    end
end

function (dataset::Dataset)(idx::Int64)
	getobs(dataset, idx)
end