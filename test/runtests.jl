using DataFrames
using FastAI
using Test

include("../src/transforms.jl")

macro excludeTest(func)
    println("$func excluded from test run")
    return nothing
end

macro includeTest(func)
    return func
end

function test_transforms_process_files()
    println("starting tests for process_files()...")
    @test process_files(".", [".dataset.jl", "test.jl"], ".jl") == ["./test.jl"]
    @test process_files(".", ["dataset.jl", "test.jl"], ".jl") == ["./dataset.jl", "./test.jl"]
    @test process_files(".", ["dataset.jl", "test.jl"]) == ["./dataset.jl", "./test.jl"]
    println("..End of tests for process_files(). Printing Summary...")
end


function test_transforms_get_files()
    println("starting tests for process_files()...")
    path = "test/test_folder"
    extension = ".txt"
    @test get_files(path,
                    extension) == ["test/test_folder/test1/test1-1.txt",
                                  "test/test_folder/test1/test1-2.txt",
                                  "test/test_folder/test2/test2-1.txt",
                                  "test/test_folder/test2/test2-2.txt"]
    println("..End of tests for process_files(). Printing Summary...")

    @test get_files(path) == ["test/test_folder/test1/test1-1.txt",
                              "test/test_folder/test1/test1-2.txt",
                              "test/test_folder/test1/test1-3.yml",
                              "test/test_folder/test2/test2-1.txt",
                              "test/test_folder/test2/test2-2.txt"]

    @test get_files(path,".yml") == ["test/test_folder/test1/test1-3.yml"]
end

function test_Filegetter()
    filegetter= FileGetter("test",".yml")
    filegetter2 = FileGetter("",".yml")
    @test filegetter("test_folder") == ["test/test_folder/test1/test1-3.yml"]
    @test_throws ErrorException filegetter2("") == "A path must be provided"
end

function test_get_image_files()
    @test get_image_files("test/test_img_folder") == ["test/test_img_folder/test.jpg"]
end

function test_Imagegetter()
    imagegetter= FileGetter("test",".jpg")
    imagegetter2 = FileGetter("",".jpg")
    @test imagegetter("test_img_folder") == ["test/test_img_folder/test.jpg"]
    @test_throws ErrorException imagegetter2("") == "A path must be provided"
end

function test_get_text_files()
    @test get_text_files("test/test_folder") == ["test/test_folder/test1/test1-1.txt",
                                                 "test/test_folder/test1/test1-2.txt",
                                                 "test/test_folder/test2/test2-1.txt",
                                                 "test/test_folder/test2/test2-2.txt"]
end

function test_Itemgetter()
    data= [(5, 4), (4, 6)]
    @test ItemGetter(1, data) == (5,4)
    
end

function test_RandomSplitter()
    X = rand(2, 6)
    split = RandomSplitter()
    train, test = split(X)
    @test length(train) == length(X)-length(test)
end

function test_TrainTestSplitter()
    y = [:a, :a, :a, :a, :a, :a, :b, :b, :b]
    split = TrainTestSplitter(y)
    train1,test1 = split(0.7)
    @test length(train1) == length(y)-length(test1)
end


function test_TrainTestSplitter2()
    X = [1 0; 1 0; 1 0; 1 0; 0 1; 0 1]
    split = TrainTestSplitter(argmax, X)
    train1,test1 = split(0.7)
    @test length(train1) == length(X)-length(test1)
end

function test_IndexSplitter()
    a,b = IndexSplitter([1,2,3,4,5], [3,5])
    @test a == [1,2,4]
    @test b == [3,5]
end

function test_grandparent_idxs()
    files = ["test/test_folder1/test2/test1-1.txt",
      "test/test_folder2/test1/test1-2.txt",
      "test/test_folder/test2/test2-1.txt",
      "test/test_folder/test2/test2-2.txt"]

    @test grandparent_idxs(files, "test2") == [1,3,4]
    @test grandparent_idxs(files, ("test1", "test2")) == [2,1,3,4]
    
end

function test_ColSplitter()
    data = DataFrame(a = [1, 2, 3], b = [true, false, true])
    split_with_name = ColSplitter(col = :b) 
    split_with_index = ColSplitter()
    a,b = split_with_name(data)
    @test a == [2]
    @test b == [1,3]
end

function test_assertBounds()
    @test assertBounds(0.3, 0,1) === nothing
    @test_throws ErrorException assertBounds(24,0,1)
end

function test_RandomSubsetSplitter()
    items = Array(1:100)
    splits = RandomSubsetSplitter(0.3, 0.1)
    trainidxs, valididxs = splits(items)
    @test length(trainidxs) === 30
    @test length(valididxs) === 11    
end

function test_parent_label()
    fnames = ["fastai_dev/dev/data/mnist_tiny/train/3/9932.png",
               "fastai_dev/dev/data/mnist_tiny/valid/4/9932.png"]
    @test parent_label("fastai_dev/dev/data/mnist_tiny/train/3/9932.png") === '3'
    @test [parent_label(path) for path in fnames] == ['3', '4']  
end

function test_RegexLabeller()
    path= "fastai_dev/dev/data/mnist_tiny/train/3/9932.png"
    pat = r"/[\d]+/"
    split = RegexLabeller(pat)
    @test split(path) == "3"
end

#todo: separate these tests
function test_ColReader()
    df = DataFrame(A = ["a", "b", "c"], B = ["1 2", "0", ""], C=["a", "b", "c"])
    colreader1 = ColReader(:A; pref="1", suff="2")
    @test colreader1(df) == ["1a2", "1b2", "1c2"]
    delimsplitter = ColReader(:B, " ")
    @test delimsplitter(df) == [["1", "2"], ["0"], [""]]
    colreader2 = ColReader([:A, :C], pref="1", suff="2")
    @test colreader2(df) == [["1a2", "1a2"], ["1b2", "1b2"], ["1c2", "1c2"]]
    colreader3 = ColReader(1, pref="1", suff="2")
    @test colreader3(df) == ["1a2", "1b2", "1c2"] 
    colreader4 = ColReader(1)
    @test colreader4(df) == ["a", "b", "c"]
end

function test_CategoryMap_default()
    df = DataFrame(A = ["d", "c", "b", "d"])
    map = CategoryMap(df[!, :A])
    @test map.items == ["b", "c", "d"]
    @test map.o2i == Dict("b" => 1, "c" => 2, "d" => 3)
end

function test_CategoryMap_add_na()
    df = DataFrame(A = ["d", "c", "b", "d"])
    map = CategoryMap(df[!, :A], add_na=true)
    @test map.items == ["#na#", "b", "c", "d"]
    @test map.o2i == Dict("#na#" => 1, "b" => 2, "c" => 3, "d" => 4)
end

function test_CategoryMap_categorical()
    df = DataFrame(A = ["d", "c", "b", "d"])
    col = categorical(df[!, :A])
    map = CategoryMap(col, add_na=true)
    @test map.items == ["#na#", "b", "c", "d"]
    @test map.o2i == Dict("#na#" => 1, "b" => 2, "c" => 3, "d" => 4)
end

function test_Categorize()
    data = ["cat", "dog", "cat"]
    cat = Categorize(data)
    @test encode(cat) == ["cat", "dog"]
    @test decode(cat, "cat") == 1
    @test decode("cat") == "cat"
    @test decode(cat, 2) == "dog"
end


 @testset "All" begin
    @excludeTest test_transforms_process_files()
    @excludeTest test_transforms_get_files()
    @excludeTest test_Filegetter()
    @excludeTest test_get_image_files()
    @excludeTest test_Imagegetter()
    @excludeTest test_get_text_files()
    @excludeTest test_Itemgetter()
    @excludeTest test_RandomSplitter()
    @excludeTest test_TrainTestSplitter2()
    @excludeTest test_IndexSplitter()
    @excludeTest test_grandparent_idxs()
    @excludeTest test_ColSplitter()
    @excludeTest test_assertBounds()
    @excludeTest test_RandomSubsetSplitter()
    @excludeTest test_parent_label()
    @excludeTest test_RegexLabeller()
    @excludeTest test_ColReader()
    @excludeTest test_CategoryMap_default()
    @excludeTest test_CategoryMap_add_na()
    @excludeTest test_CategoryMap_categorical()
    @includeTest test_Categorize()
end
