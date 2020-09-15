### A Pluto.jl notebook ###
# v0.11.13

using Markdown
using InteractiveUtils

# ╔═╡ a2850860-f73b-11ea-159a-ad5bc1704c57
using Images, FileIO

# ╔═╡ 7302a090-f739-11ea-3fd7-2786faf0a7bd
md"## Data utils
Contains helper functions to deal with data extraction, download etc.

**TODO**
Functions for downloading, saving and other actions are pending"

# ╔═╡ cad848b0-f739-11ea-0343-177efd090dc6
md"Many functions from `Datadeps` package can be extended for data handeling.
One such function is `Unpack`.

##### unpack
`Unpack` takes the archive name and uncompresses it. Optionally a destination path can be provided or by default the uncompression is done in the present working directory. The archive file gets deleted by default once the files are extracted. However, this can be changed."

# ╔═╡ 98ef39c0-f73a-11ea-0426-dda23207aeb9
begin
import DataDeps: unpack_cmd, unpack
	
function unpack(f; filepath=pwd(), keep_originals=false)
        run(unpack_cmd(f, filepath, last(splitext(f)), last(splitext(first(splitext(f))))))
        rm("pax_global_header"; force=true)
        !keep_originals && rm(f)
end
	
end

# ╔═╡ c9e2a8a0-f73a-11ea-11a1-55fa9f327373
md"This can then be used to extract a compressed file."

# ╔═╡ d8a383a0-f73a-11ea-3fbd-cf9783b8ad3d
#unpack(<path to the archive file, filepath=<destination path>)

# ╔═╡ 2985b0e0-f73b-11ea-0a83-c52b1117dece
md"##### load_image
Reads an image file and loads it to memory.

It is a thin wrapper over the Images.load functions and is more of a convenience function.
"

# ╔═╡ a01c93e0-f73b-11ea-3e76-4fe742ac8301
function load_image(path::AbstractString)
	if isfile(path)
		return load(path)
	end
end

# ╔═╡ d99a2330-f73b-11ea-0fa1-fbd0524c58ca
md"##### image2array

A thin wrapper over `Images.channelview` function. convenience function to get the fixed poitn array of an image"

# ╔═╡ 129a4520-f73c-11ea-1272-cbe8a3ca2f08
function image2array(image)
	channelview(image)
end

# ╔═╡ Cell order:
# ╟─7302a090-f739-11ea-3fd7-2786faf0a7bd
# ╟─cad848b0-f739-11ea-0343-177efd090dc6
# ╠═98ef39c0-f73a-11ea-0426-dda23207aeb9
# ╟─c9e2a8a0-f73a-11ea-11a1-55fa9f327373
# ╠═d8a383a0-f73a-11ea-3fbd-cf9783b8ad3d
# ╟─2985b0e0-f73b-11ea-0a83-c52b1117dece
# ╠═a2850860-f73b-11ea-159a-ad5bc1704c57
# ╠═a01c93e0-f73b-11ea-3e76-4fe742ac8301
# ╟─d99a2330-f73b-11ea-0fa1-fbd0524c58ca
# ╠═129a4520-f73c-11ea-1272-cbe8a3ca2f08
