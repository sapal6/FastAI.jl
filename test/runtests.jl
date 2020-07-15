using FastAI2Julia
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

function test_RangeNumber()
    @test RangeNumber(3, 1:7) == RangeNumber{Int64,1:7}(3)
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
    @includeTest test_RangeNumber()
end