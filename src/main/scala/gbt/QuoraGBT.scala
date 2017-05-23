package gbt

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

/**
  * Created by limingming on 2017/4/18.
  */
object QuoraGBT {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().appName("randomForest").getOrCreate()

    val document = spark.read.format("csv").load("/random/train.csv")

    val tokenizer1 = new Tokenizer().setInputCol("_c3").setOutputCol("words1")
    val tokenizer2 = new Tokenizer().setInputCol("_c4").setOutputCol("words2")

    val tokenized1 = tokenizer1.transform(document)
    val tokenized2 = tokenizer2.transform(tokenized1)

    val word2Vec1 = new Word2Vec().setInputCol("words1").setOutputCol("vec1").setVectorSize(3).setMinCount(0)
    val word2Vec2 = new Word2Vec().setInputCol("words2").setOutputCol("vec2").setVectorSize(3).setMinCount(0)

    val vector1 = word2Vec1.fit(tokenized2).transform(tokenized2)
    val vector2 = word2Vec2.fit(vector1).transform(vector1)

    val assembler = new VectorAssembler().setInputCols(Array("vec1", "vec2")).setOutputCol("features")
    val vector = assembler.transform(vector2)

    val labelIndexer = new StringIndexer().setInputCol("_c5").setOutputCol("indexedLabel").fit(vector)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4) fit (vector)


    //
    val Array(trainingData, testData) = vector.randomSplit(Array(0.7, 0.3))

    val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(100)

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]


    println("Learned classification forest model:\n" + gbtModel.toDebugString)

  }
}
