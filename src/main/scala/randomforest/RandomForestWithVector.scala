package randomforest


import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{VectorIndexer, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.sql._

/**
  * Created by limingming on 2017/5/4.
  */
object RandomForestWithVector {

  def main(args: Array[String]) {
    val spark: SparkSession = SparkSession.builder().appName("randomForest").getOrCreate()
    val conf: SparkConf = new SparkConf().setMaster("local").setAppName("test word2vec ")
    val sc: SparkContext = new SparkContext(conf)


    val train = sc.textFile("/raondom2/trainVec.csv") //load train data

    val trainSplit = train.map(_.split(",")) //split string with comma
    val trainFilter: RDD[Array[String]] = trainSplit.filter(!_.contains("id")) //substract the head
    val trainDF = spark.createDataFrame(trainFilter.map { case Array(a0, a1, a2, a3, a4, a5, a6) => (a0, a1, a2, a3, Vectors.dense(a4.split("  ").map(_.toDouble)), Vectors.dense(a5.split("  ").map(_.toDouble)), a6) }).toDF() //convert to dataframe

    val trainAssembler = new VectorAssembler().setInputCols(Array("_5", "_6")).setOutputCol("features")
    val trainVector = trainAssembler.transform(trainDF) // assemble the two vectors

    val trainLabelIndexed = new StringIndexer().setInputCol("_7").setOutputCol("indexedLabel").fit(trainVector).transform(trainVector) //index label
    val trainFeatureIndexed = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(trainLabelIndexed).transform(trainLabelIndexed) //index feature

    val rfModel = new RandomForestRegressor().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(300).fit(trainFeatureIndexed) //train to get the model

    rfModel.save("/random2/model/rfModel-400d-100t")

    val test = sc.textFile("/random2/testVec.csv")

    val testSplit = test.map(_.split(","))
    val testFilter = testSplit.filter(!_.contains("test_id"))
    val testDF = spark.createDataFrame(testFilter.map { case Array(a0, a1, a2, a3) => (a0, a1, Vectors.dense(a2.split("  ") map (_.toDouble)), Vectors.dense(a3.split("  ") map (_.toDouble))) }).toDF()

    val testAssembler = new VectorAssembler().setInputCols(Array("_3", "_4")).setOutputCol("features")
    val testVector = testAssembler.transform(testDF)

    val testLabelIndexed = new StringIndexer().setInputCol("_1").setOutputCol("indexedLabel").fit(testVector).transform(testVector)
    val testFeatureIndexed = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(testLabelIndexed).transform(testLabelIndexed)

    val predictions: DataFrame = rfModel.transform(testFeatureIndexed)
    predictions.select("_1", "prediction").rdd.saveAsTextFile("/random2/prediction/sample2")

  }

}
