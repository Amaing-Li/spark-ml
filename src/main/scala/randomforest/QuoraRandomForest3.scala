package randomforest


import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature._
import org.apache.spark.sql._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by limingming on 2017/4/20.
  */
object QuoraRandomForest3 {

  def main(args: Array[String]) {
    //创建入口
    val spark = SparkSession.builder().appName("randomForest").getOrCreate()

    val document: DataFrame = spark.read.format("csv").load("/random/train.csv")

    //将句子分成单词
    val trainTokenized1 = new Tokenizer().setInputCol("_c3").setOutputCol("words1").transform(document)
    val trainTokenized2 = new Tokenizer().setInputCol("_c4").setOutputCol("words2").transform(trainTokenized1)

    //创建视图，进行暴力数据清洗（去除格式不对的数据）
    trainTokenized2.createOrReplaceTempView("train")
    val trainCleaned = spark.sql("select * from train where (_c5 == '0' or _c5 == '1') and _c0 != null and _c1 != null and _c2 != null and _c3 != null and _c4 != null")


    //转成向量
    val trainVector1 = new Word2Vec().setInputCol("words1").setOutputCol("vec1").setVectorSize(200).setMinCount(0).fit(trainCleaned).transform(trainCleaned)
    val trainVector2 = new Word2Vec().setInputCol("words2").setOutputCol("vec2").setVectorSize(200).setMinCount(0).fit(trainVector1).transform(trainVector1)

    //向量的合并
    val trainAssembler = new VectorAssembler().setInputCols(Array("vec1", "vec2")).setOutputCol("features")
    val trainVector = trainAssembler.transform(trainVector2)

    //添加索引
    val trainLabelIndexed = new StringIndexer().setInputCol("_c5").setOutputCol("indexedLabel").fit(trainVector).transform(trainVector)
    val trainFeatureIndexed = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit (trainLabelIndexed).transform(trainLabelIndexed)


    val rfModel = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(200).fit(trainFeatureIndexed)

    //val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)



    //start predict

    val test: DataFrame = spark.read.format("csv").load("/random/test.csv")

    //句子分成单词
    val testTokenized1 =  new Tokenizer().setInputCol("_c1").setOutputCol("words1").transform(test)
    val testTokenized2 =  new Tokenizer().setInputCol("_c2").setOutputCol("words2").transform(testTokenized1)

    //数据清洗
    testTokenized2.createOrReplaceTempView("test")
    val testCleaned = spark.sql("select * from test where _c0 != null and _c1 != null and _c2 != null")

    //转成向量
    val testVector1 =  new Word2Vec().setInputCol("words1").setOutputCol("vec1").setVectorSize(200).setMinCount(0).fit(testCleaned).transform(testCleaned)
    val testVector2 = new Word2Vec().setInputCol("words2").setOutputCol("vec2").setVectorSize(200).setMinCount(0).fit(testVector1).transform(testVector1)

    //合成一个向量
    val testAssembler = new VectorAssembler().setInputCols(Array("vec1", "vec2")).setOutputCol("features")
    val testVector = testAssembler.transform(testVector2)

    //添加索引
    val testLabelIndexed = new StringIndexer().setInputCol("_c0").setOutputCol("indexedLabel").fit(testVector).transform(testVector)
    val testFeatureIndexed = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit (testLabelIndexed).transform(testLabelIndexed)

    val predictions = rfModel.transform(testFeatureIndexed)


    println("Learned classification forest model:\n" + rfModel.toDebugString)
  }
}
