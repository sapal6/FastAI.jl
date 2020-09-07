abstract type Transform end
#TODO:code is repeated for error
#TODO: create reusable func
function encode(item) 
    if item===nothing
        error("A Transform must be provided")
    end
    item
end

function decode(item)
    if item===nothing
        error("A Transform must be provided to decode")
    end
    item
end