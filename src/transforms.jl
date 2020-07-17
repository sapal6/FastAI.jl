
#=
Transforms.jl:

Author: Satyabrata pal (satyabrata.pal1@gmail.com)

Acknowledgements-
Original source -
    https://github.com/fastai/fastai2/blob/master/fastai2/data/transforms.py

Original documentation-
    https://github.com/fastai/fastai2/blob/master/nbs/05_data.transforms.ipynb

Helper functions for processing data and basic transforms
    Functions for getting, splitting, and labeling data, 
    as well as generic transforms

Get, split, and label
    For most data source creation we need functions to get
    a list of items, split them in to train/valid sets, and
    label them.
=#
using Random
using Unicode
using MLDataUtils
using DataStructures
using DataFrames

#TODO: ignore case while checking extesnions
function process_files(path::AbstractString,
                       files::AbstractArray,
                        extensions=nothing)
    #gets file as per an extension
      res = []
      extensions === nothing ? res = [joinpath(path, file)
                                    for file in files
                                         if .!startswith(file, ".")] :
                                      res = [joinpath(path, file) for file in files if .!startswith(file, ".") 
                                      &&
                                       occursin(extensions, Unicode.normalize(file, casefold=true))]
      res
end


#todo: figure out what is teh role of the "folders" parameter in the original code
function get_files(path::AbstractString,
    extensions=nothing,
    recurse::Bool=true)

    if length(path) == 0
        error("A path must be provided")
    end

    #Get all the files in path with optional extensions, optionally with recurse, only in folders, if specified.
    res = AbstractString[]
    file_names = []

    if recurse
       for (root, dirs, files) in walkdir(path)
           for file in files
               push!(file_names, joinpath(root,file))
           end
       end
       res = process_files("", file_names, extensions)
    else
       files = [file for file in readdir(path, join=true) if isfile(file)]
       res = process_files(path, files, extensions)
    end
    res
end


#=
Curry function that will provide the arguments first and then
wait for the pathsuffix later

Example from - https://riptutorial.com/julia-lang/example/20261/implementing-currying

It's often useful to be able to create functions with customized behavior. fastai.
data generally uses functions named as CamelCase verbs ending in er to create these 
functions. FileGetter is a simple example of such a function creator.
e.g. const filegetter= FileGetter("path",".csv")
filegetter("/test")
=#


function FileGetter(path, extensions=nothing, recurse=true)
    #Create `get_files` partial function that searches path suffix `suf`,
    #only in `folders`, if specified, and passes along args
     pathsuffix -> get_files(joinpath(path, pathsuffix),extensions, recurse)
end

#=
"Get image files in `path` recursively, only in `folders`, if specified."
Convienience function to get images with standard image extension
=#
function get_image_files(path, recurse=true, folders=nothing)
    res = []
    image_extensions=Set{AbstractString}(["tiff", "jpeg", "png", "gif", "jpg"])
    foreach(image_extensions) do img_extension
        res = [file for file in get_files(path, img_extension, recurse)]
    end
    res
end

#=
Curry function that will provide the arguments first and then
wait for the pathsuffix later

Example from - https://riptutorial.com/julia-lang/example/20261/implementing-currying

e.g. const imagegetter= ImageGetter("path",".csv")
filegetter("/test")
=#
function ImageGetter(path, recurse=true, folders=nothing)
    "Create `get_image_files` partial function that searches path suffix `suf`
     and passes along `kwargs`, only in `folders`, if specified."
     pathsuffix -> get_image_files(joinpath(path, pathsuffix), recurse, folders)
end

#=
Helper for text files
=#
function get_text_files(path, recurse=true, folders=nothing)
    "Get text files in `path` recursively, only in `folders`, if specified."
    get_files(path, ".txt", recurse)
end

#=
Accessing items across tuples when a specific
index is provided.

Soft implementation of ItemGetter in the original source-
  https://github.com/fastai/fastai2/blob/master/fastai2/data/transforms.py

Helpful for collecting labels of inputs.
e.g. datatuple= [(5, 4), (4, 6)]
          ItemGetter(1)
          output-- (5,4)
=#
function ItemGetter(index::Integer, x)
   Tuple( (a->a[index]).(x))
end

#=Does it make sense to design somthing like this in Julia?
Accessing fields across tuples when a specific
index is provided.

implementation of AttrGetter in the original source-
  https://github.com/fastai/fastai2/blob/master/fastai2/data/transforms.py

=#
#=original code
class AttrGetter(ItemTransform):
    "Creates a proper transform that applies `attrgetter(nm)` (even on a tuple)"
    _retain = False
    def __init__(self, nm, default=None): store_attr(self, 'nm,default')
    def encodes(self, x): return getattr(x, self.nm, self.default)
=#


#=
Split
The next set of functions are used to split data into training and validation sets.
The functions return two lists - a list of indices or masks for each of training 
and validation sets.

These APIs cloesly mimic the original code's APIs.

Functions from the MLDataUtils are leveraged here.
=#

#=Wrapper around the splitobs function from here -
http://mldatapatternjl.readthedocs.io/en/latest/documentation/datasubset.html#split

e.g. X = rand(2, 6)
     split = RandomSplitter()
     train, test = split(X)
     --julia> train
              2×4 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
               0.226582  0.933372  0.505208   0.0443222
               0.504629  0.522172  0.0997825  0.722906

       julia> test
       2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
        0.812814  0.11202
        0.245457  0.000341996

=#
function RandomSplitter(valid_pct=0.2)
    datatable -> splitobs(datatable, at = valid_pct);
end

#=
Wrapper around stratifiedobs--
http://mldatapatternjl.readthedocs.io/en/latest/documentation/targets.html#stratified

Can be used for stratified partioning of data
=#

function TrainTestSplitter(data)
     (args...) -> stratifiedobs(data, args...)
end

function TrainTestSplitter(f,data)
    (args...) -> stratifiedobs(f,data, args...)
end

#=
Split `items` so that `val_idx` are in the validation set 
and the others in the training set
=#
function IndexSplitter(items::AbstractArray, valid_idx::AbstractArray)
    train_idx= setdiff(SortedSet(items),SortedSet(valid_idx))
    collect(train_idx), collect(valid_idx)
end

#=
Return an array of indices of parent directories of files
e.g. grandparent_idxs(["/folder/train/test.png", "/folder/valid/test2.png"],
                      "train")
[1]
=#
#the bool array is exactly len of actual array * len of names
#have to find a way to find the true indices

#TODO: code can be much cleaner
function grandparent_idxs(items::AbstractArray, name::AbstractString)
    truthvalues = [occursin("/$name/", item) for item in items]
    idxs=findall(truthvalues)
    idxs
end
#TODO: code can be much cleaner
function grandparent_idxs(items::AbstractArray, names::Tuple)
    truths=[]
    for name in names
        truthvalues = []
        for item in items
            push!(truthvalues,occursin("/$name/", item))
        end
        push!(truths, truthvalues)
    end
    idxs=[idx for truth in truths for idx in findall(truth)]
    idxs
end

#=
Split `items` from the grand parent folder names (`train_name` and `valid_name`).
e.g. fnames -> GrandparentSplitter(train_name='train', valid_name='valid')
[]
=#
function GrandparentSplitter()
    train_idxs = items -> grandparent_idxs(items, "train")
    valid_idxs = items -> grandparent_idxs(items, "valid")
end

function GrandparentSplitter(train_name::AbstractString, valid_name::AbstractString)
    train_idxs = items -> grandparent_idxs(items, train_name)
    valid_idxs = items -> grandparent_idxs(items, valid_name)
end

function GrandparentSplitter(train_name::Tuple, valid_name::Tuple)
    train_idxs = items -> grandparent_idxs(items, train_name)
    valid_idxs = items -> grandparent_idxs(items, valid_name)
end

#=
do we need to port the below functions?
In juli acreating anonymous functions are easier than in python.
Keeping this in mind do we need a top level API to split indices 
on basis of a fucntion output?

Similar question applies for the MaskSplitter function and FileSplitter.
=#
#= def FuncSplitter(func):
    "Split `items` by result of `func` (`True` for validation, `False` for training set)."
    def _inner(o, **kwargs):
        val_idx = mask2idxs(func(o_) for o_ in o)
        return IndexSplitter(val_idx)(o)
    return _inner

# export
def MaskSplitter(mask):
    "Split `items` depending on the value of `mask`."
    def _inner(o, **kwargs): return IndexSplitter(mask2idxs(mask))(o)
    return _inner 
    
def FileSplitter(fname):
    "Split `items` depending on the value of `mask`."
    valid = Path(fname).read().split('\n')
    def _func(x): return x.name in valid
    def _inner(o, **kwargs): return FuncSplitter(_func)(o)
    return _inner
    =#

#=
"Split `items` (supposed to be a dataframe) by value in `col`"
e.g df = {'a': [0,1,2,3,4], 'b': [True,False,True,True,False]}
split = ColSplitter("b")
split(df)
[[2,5],[1,3,4]]
=#
function ColSplitter(;col=:is_valid)
    df -> IndexSplitter([DataFrames.row(collect(eachrow(df))[i])
                          for i in 1:size(df, 1)],
                          findall(df[col])) 
end

function ColSplitter(col::Integer)
    df -> IndexSplitter([DataFrames.row(collect(eachrow(df))[i])
                            for i in 1:size(df, 1)],
                            findall(df[!, col]))     
end

#= checking the bounds - =#
function assertBounds(num, lowerbound, upperbound)
    lowerbound<num<upperbound ? nothing : error("$num in
                                          $lowerbound:$upperbound")
end

#=
Take randoms subsets of `splits` with `train_sz` and `valid_sz`
e.g items= [1,2,3,4,5,6,7,8,9,10]
RandomSubsetSplitter(0.3,0.1)
=#

#= TODO: does anythng already exists in the ecosystem which does 
something similar to this but in much more efficient way=#
function Splitter(items, train_sz, valid_sz)
    train_valid_len  = (convert(Int,round(length(items)*train_sz)),
                         convert(Int,round(length(items)*valid_sz)))
    idxs = shuffle!([i for i in 1:length(items)])
    idxs[1:train_valid_len[1]],idxs[train_valid_len[1]:train_valid_len[1]+
                       train_valid_len[2]]
end

function RandomSubsetSplitter(train_sz, valid_sz)
    assertBounds(train_sz, 0,1)
    assertBounds(valid_sz, 0,1)
    items -> Splitter(items, train_sz, valid_sz)
end

#=
Label -
 The final set of functions is used to label a single item of data.
=#

#=
"Label `item` with the parent folder name."
e.g. parent_label("fastai_dev/dev/data/mnist_tiny/train/3/9932.png"))
3
=#
function parent_label(path::AbstractString, args...)
    last(dirname(path))
end
#= Label `item` with regex `pat`. 
e.g. pattern = RegexLabeller("fastai_dev/dev/data/mnist_tiny/train/3/9932.png")
     pattern("[0-9]+")
     '3'
=#
function RegexLabeller(pat)
    path -> strip(convert(String,(match(pat, "$path")).match), ['/', '/'])
end

