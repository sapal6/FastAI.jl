
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

using MLDataUtils

#TODO: ignore case while checking extesnions
function process_files(path, files, extensions=nothing)
    #gets file as per an extension
    #todo: create the actual fucntion
    res = [joinpath(path, file) for file in files if !startswith(file, ".") && ((!extensions) || occursin(".$extensions", file))]
    res
end

#TODO: "folders"args is not used yet.
function get_files(path::AbstractString,
             extensions=nothing,
             recurse::Bool=true,
             folders=nothing)

         """Get all the files in `path` with optional `extensions`, optionally with `recurse`, only in `folders`, if specified."""

         if recurse
            res = AbstractString[]
            root, dirs, files = walkdir(path)
            files = [joinpath(root,file) for file in files]
            push!(res, process_files(path, files, extensions))
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


function FileGetter(path, extensions=nothing, recurse=true, folders=nothing)
    "Create `get_files` partial function that searches path suffix `suf`,
     only in `folders`, if specified, and passes along args"
     pathsuffix -> get_files(joinpath(path, pathsuffix),extensions, recurse, folders)
end

#=
"Get image files in `path` recursively, only in `folders`, if specified."
Convienience function to get images with standard image extension
=#
function get_image_files(path, recurse=true, folders=nothing)
    image_extensions=Set{AbstractString}(["TIFF", "JPEG", "PNG", "GIF"])
    get_files(path, image_extensions, recurse, folders)
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
    get_files(path, [".txt"], recurse, folders)
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