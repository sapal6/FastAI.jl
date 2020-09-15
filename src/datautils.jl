using Images, File
import DataDeps: unpack_cmd, unpack

#TODO: not returning extracted path
#TODO: performance is slow and sometimes progress is not visible
#TODO: tests pending
function unpack(f; filepath=pwd(), keep_originals=false)
    run(unpack_cmd(f, filepath, last(splitext(f)), last(splitext(first(splitext(f))))))
    rm("pax_global_header"; force=true)
    !keep_originals && rm(f)
end

#=load_image
read an image file and load it to memory

It is a wrapper over the Images.load functions=#
function load_image(path::AbstractString)
	if isfile(path)
		return load(path)
	end
end

#=
image2array
convert image to it's array representation

it is a wrapper over the Images.channelview function=#
function image2array(image)
	convert(Array{Float32}, channelview(image))
end