package randomforest

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover


/**
  * Created by limingming on 2017/4/17.
  */
object QuoraRandomForest {
  def main(args: Array[String]) {

    val spark = SparkSession.builder().appName("randomForest").getOrCreate()

    val document = spark.read.format("csv").load("/random/rt/train.csv")

    val tokenizer1 = new Tokenizer().setInputCol("_c3").setOutputCol("words1")
    val tokenizer2 = new Tokenizer().setInputCol("_c4").setOutputCol("words2")

    val tokenized1 = tokenizer1.transform(document)
    val tokenized2 = tokenizer2.transform(tokenized1)

    tokenized2.createOrReplaceTempView("train")
    val trainCleaned = spark.sql("select * from train where (_c5 == '0' or _c5 == '1') and _c0 != null and _c1 != null and _c2 != null and _c3 != null and _c4 != null")

    val stopWordsRemoved1 = new StopWordsRemover().setInputCol("words1").setOutputCol("filtered1").transform(trainCleaned)
    val stopWordsRemoved2 = new StopWordsRemover().setInputCol("words2").setOutputCol("filtered2").transform(stopWordsRemoved1)

    val word2Vec1 = new Word2Vec().setInputCol("filtered1").setOutputCol("vec1").setVectorSize(10).setMinCount(0)
    val word2Vec2 = new Word2Vec().setInputCol("filtered2").setOutputCol("vec2").setVectorSize(10).setMinCount(0)

    val vector1 = word2Vec1.fit(stopWordsRemoved2).transform(stopWordsRemoved2)
    val vector2 = word2Vec2.fit(vector1).transform(vector1)

    val assembler = new VectorAssembler().setInputCols(Array("vec1", "vec2")).setOutputCol("features")
    val vector = assembler.transform(vector2)

    val labelIndexer = new StringIndexer().setInputCol("_c5").setOutputCol("indexedLabel").fit(vector)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(vector)


    //
    val Array(trainingData, testData) = vector.randomSplit(Array(0.9, 0.1))


    val rf = new RandomForestClassifier().setLabelCol("indexedLabel") .setFeaturesCol("indexedFeatures").setNumTrees(100).setSubsamplingRate(0.5)

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))



    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]

    println("Learned classification forest model:\n" + rfModel.toDebugString)

  }

}
