package randomforest

/**
  * Created by limingming on 2017/4/22.
  */
object QuoraRandomForestRegression {
  def main(args: Array[String]) {
    val spark: SparkSession = SparkSession.builder().appName("randomForest").getOrCreate()
    val conf = new SparkConf().setMaster("local").setAppName("test word2vec ")
    val sc = new SparkContext(conf)


    /**
      * 用语料库创建模型
      */
    val dataPath = "/random/all561.txt"
    val input = sc.textFile(dataPath).map(line => line.split(" ").toSeq)
    val documentDF = spark.createDataFrame(input.map(Tuple1.apply)).toDF("text")

    documentDF.createOrReplaceTempView("document")
    val documentDFCleaned = spark.sql("select * from document where text != null")


    val word2Vec: Word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(200).setMinCount(0)
    val word2VecModel: Word2VecModel = word2Vec.fit(documentDF)


    word2VecModel.save("/random/model/word2VecModel200")

    //val word2VecModel = Word2VecModel.load("/random/model/word2VecModel")





    /**
      * 利用训练的语料库转向量
      */
    val trainData = spark.read.format("csv").load("/random/rt/train.csv")

    //将句子分成单词
    val trainTokenized1 = new Tokenizer().setInputCol("_c3").setOutputCol("words1").transform(trainData)
    val trainTokenized2 = new Tokenizer().setInputCol("_c4").setOutputCol("words2").transform(trainTokenized1)

    //创建视图，进行暴力数据清洗（去除格式不对的数据）
    trainTokenized2.createOrReplaceTempView("train")
    val trainCleaned = spark.sql("select * from train where (_c5 == '0' or _c5 == '1') and _c0 != null and _c1 != null and _c2 != null and _c3 != null and _c4 != null")


    val stopWordsRemoved1 = new StopWordsRemover().setInputCol("words1").setOutputCol("filtered1").transform(trainCleaned)
    val stopWordsRemoved2 = new StopWordsRemover().setInputCol("words2").setOutputCol("filtered2").transform(stopWordsRemoved1)


    //转成向量
    val trainVector1 = word2VecModel.setInputCol("filtered1").setOutputCol("vec1") transform (stopWordsRemoved2)
    val trainVector2 = word2VecModel.setInputCol("filtered2").setOutputCol("vec2").transform(trainVector1)

    //向量的合并
    val trainAssembler = new VectorAssembler().setInputCols(Array("vec1", "vec2")).setOutputCol("features")
    val trainVector = trainAssembler.transform(trainVector2)



    /**
      * 训练随机森林
      */
    //添加索引
    val trainLabelIndexed = new StringIndexer().setInputCol("_c5").setOutputCol("indexedLabel").fit(trainVector).transform(trainVector)
    val trainFeatureIndexed = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(trainLabelIndexed).transform(trainLabelIndexed)

    val rfModel = new RandomForestRegressor().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(100).setSubsamplingRate(0.5).fit(trainFeatureIndexed)
    //rfModel.save("/random/model/rfModelWithSpace")


    /**
      * 利用训练的随机森林进行预测
      */
    val test: DataFrame = spark.read.format("csv").load("/random/rt/test.csv")

    //句子分成单词
    val testTokenized1 = new Tokenizer().setInputCol("_c1").setOutputCol("words1").transform(test)
    val testTokenized2 = new Tokenizer().setInputCol("_c2").setOutputCol("words2").transform(testTokenized1)

    //数据清洗
    testTokenized2.createOrReplaceTempView("test")
    val testCleaned = spark.sql("select * from test where _c0 != null and _c1 != null and _c2 != null")

    //转成向量
    val testVector1 = word2VecModel.setInputCol("words1").setOutputCol("vec1").transform(testCleaned)
    val testVector2 = word2VecModel.setInputCol("words2").setOutputCol("vec2").transform(testVector1)

    //合成一个向量
    val testAssembler = new VectorAssembler().setInputCols(Array("vec1", "vec2")).setOutputCol("features")
    val testVector = testAssembler.transform(testVector2)

    //添加索引
    val testLabelIndexed = new StringIndexer().setInputCol("_c0").setOutputCol("indexedLabel").fit(testVector).transform(testVector)
    val testFeatureIndexed = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(testLabelIndexed).transform(testLabelIndexed)

    val predictions: DataFrame = rfModel.transform(testFeatureIndexed)

    predictions.select("_c0", "prediction").rdd.saveAsTextFile("/random/result/sample2")




    //    val vecs = model.getVectors
    //    vecs.show
    //    vecs.rdd.saveAsTextFile("/random/DATA")

  }

}
