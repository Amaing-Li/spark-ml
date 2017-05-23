package decisiontree

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession

/**
  * Created by limingming on 2017/5/23.
  */
object QuoraDecisionTree {

  def main(args: Array[String]) {
    //产生入口
    val spark = SparkSession.builder().appName("decisiontree").getOrCreate()

    //读取数据为DataFrame，并赋予列名为label，text1，text2
    val document = spark.createDataFrame(Seq(
      ("0", "What is the step by step guide to invest in share market in india?".split(" "), "What is the step by step guide to invest in share market?".split(" ")),
      ("0", "What is the story of Kohinoor (Koh-i-Noor) Diamond?".split(" "), "What would happen if the Indian government stole the Kohinoor (Koh-i-Noor) diamond back?".split(" ")),
      ("0", "How can I increase the speed of my internet connection while using a VPN?".split(" "), "How can Internet speed be increased by hacking through DNS?".split(" ")),
      ("0", "Why am I mentally very lonely? How can I solve it?".split(" "), "Find the remainder when [math]23^{24}[/math] is divided by 24,23?".split(" ")),
      ("0", "Which one dissolve in water quikly sugar, salt, methane and carbon di oxide?".split(" "), "Which fish would survive in salt water?".split(" ")),
      ("1", "Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me?".split(" "), "I'm a triple Capricorn (Sun, Moon and ascendant in Capricorn) What does this say about me?".split(" ")),
      ("0", "Should I buy tiago?".split(" "), "What keeps childern active and far from phone and video games?".split(" ")),
      ("1", "How can I be a good geologist?".split(" "), "What should I do to be a great geologist?".split(" ")),
      ("0", "When do you use 銈?instead of 銇?".split(" "), "When do you use \"&\" instead of \"and\"?".split(" ")),
      ("0", "Motorola (company): Can I hack my Charter Motorolla DCX3400?".split(" "), "How do I hack Motorola DCX3400 for free internet?".split(" "))
    )).toDF("label", "text1", "text2")

    //将text1转成向量，维度为3，这是一个转化器
    val word2Vec1 = new Word2Vec().setInputCol("text1").setOutputCol("vec1").setVectorSize(3).setMinCount(0)
    //将text2转成向量，维度为3，这是一个转化器
    val word2Vec2 = new Word2Vec().setInputCol("text2").setOutputCol("vec2").setVectorSize(3).setMinCount(0)

    //text转化为向量
    val vector1 = word2Vec1.fit(document).transform(document)
    val vector2 = word2Vec2.fit(document).transform(vector1)

    //向量合并，将向量vec1和vec2合并，并取列名为features，这是合并器
    val assembler = new VectorAssembler().setInputCols(Array("vec1", "vec2")).setOutputCol("features")
    //进行合并
    val vector = assembler.transform(vector2)

    //对label列建立索引
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(vector)
    //对features列建立索引
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit (vector)

    //将数据分为训练数据和测试数据
    val Array(trainingData, testData) = vector.randomSplit(Array(0.7, 0.3))

    //创建决策树生成器
    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

    //索引转化器，将索引的列反向转化
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    //连接成pipeline
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    //对训练数据进行训练，生成model
    val model = pipeline.fit(trainingData)

    //利用决策树，对测试数据进行预测
    val predictions = model.transform(testData)

    //评估
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

    //取得决策树，并打印
    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)
  }

}
