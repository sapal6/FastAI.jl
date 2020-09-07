import DataDeps: unpack_cmd, unpack

#TODO: not returning extracted path
#TODO: performance is slow and sometimes progress is not visible
#TODO: tests pending
function unpack(f; filepath=pwd(), keep_originals=false)
    run(unpack_cmd(f, filepath, last(splitext(f)), last(splitext(first(splitext(f))))))
    rm("pax_global_header"; force=true)
    !keep_originals && rm(f)
end