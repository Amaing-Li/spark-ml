package gbt

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql._
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by limingming on 2017/5/5.
  */
object GBTWithVector {
  def main(args: Array[String]) {
    val spark: SparkSession = SparkSession.builder().appName("randomForest").getOrCreate()
    val conf: SparkConf = new SparkConf().setMaster("local").setAppName("test word2vec ")
    val sc: SparkContext = new SparkContext(conf)



    val train = sc.textFile("/random3/train1000dVec-stop.csv") //load train data

    val trainSplit = train.map(_.split(",")) //split string with comma
    val trainFilter = trainSplit.filter(!_.contains("id")) //substract the head
    val trainDF = spark.createDataFrame(trainFilter.map { case Array(a0, a1, a2, a3, a4, a5) => (a0, a1, a2, Vectors.dense(a3.split("  ").map(_.toDouble)), Vectors.dense(a4.split("  ").map(_.toDouble)), a5) }).toDF() //convert to dataframe


    val trainAssembler = new VectorAssembler().setInputCols(Array("_4", "_5")).setOutputCol("features")
    val trainVector = trainAssembler.transform(trainDF) // assemble the two vectors

    val trainLabelIndexed = new StringIndexer().setInputCol("_6").setOutputCol("indexedLabel").fit(trainVector).transform(trainVector) //index label
    val trainFeatureIndexed = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(trainLabelIndexed).transform(trainLabelIndexed) //index feature

    val gbtModel = new GBTRegressor().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(100).fit(trainFeatureIndexed) //train to get the model

    //gbtModel.save("/random2/model/gbtModel-400d-300t")


    val test = sc.textFile("/random3/test1000dVec-stop.csv")

    val testSplit = test.map(_.split(","))
    val testFilter = testSplit.filter(!_.contains("test_id"))
    val testDF = spark.createDataFrame(testFilter.map { case Array(a0, a1, a2) => (a0, Vectors.dense(a1.split("  ") map (_.toDouble)), Vectors.dense(a2.split("  ") map (_.toDouble))) }).toDF()

    val testAssembler = new VectorAssembler().setInputCols(Array("_2", "_3")).setOutputCol("features")
    val testVector = testAssembler.transform(testDF)

    val testLabelIndexed = new StringIndexer().setInputCol("_1").setOutputCol("indexedLabel").fit(testVector).transform(testVector)
    val testFeatureIndexed = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(testLabelIndexed).transform(testLabelIndexed)

    val predictions: DataFrame = gbtModel.transform(testFeatureIndexed)
    predictions.select("_1", "prediction").rdd.saveAsTextFile("/random3/pred/sample1")

  }

}
