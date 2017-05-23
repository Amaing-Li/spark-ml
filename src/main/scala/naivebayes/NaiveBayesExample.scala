package naivebayes

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

/**
  * Created by limingming on 2017/5/22.
  */
object NaiveBayesExample {
  def main(args: Array[String]) {

    //spark入口
    val spark: SparkSession = SparkSession.builder().getOrCreate()

    //这是训练数据得到模型的代码
    val data = spark.read.format("csv").load("/c").toDF("category", "text") //训练数据存放的目录或文件
    //添加索引
    val indexer = new StringIndexer().setInputCol("category").setOutputCol("label").fit(data)
    val training = indexer.transform(data)
    //切成单词
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val words = tokenizer.transform(training)
    //转成features
    val hashingTF = new HashingTF().setNumFeatures(500000).setInputCol("words").setOutputCol("rawFeatures")
    val features = hashingTF.transform(words)
    //词频
    val idfModel = new IDF().setInputCol("rawFeatures").setOutputCol("features").fit(features)
    val idf = idfModel.transform(features)
    //训练得到模型
    val model = new NaiveBayes().setFeaturesCol("features").setLabelCol("label").fit(idf)
    //model.save("") //模型要保存的路径

    //下面是做预测的代码
    val test = spark.read.format("csv").load("/test10").toDF("text") //读取测试数据，格式为：距离 中国 首艘 国产 航母 001 A 型 山东 舰 下水 还 不到 一个月 的 时间 中国 互联网 上 18日 出现
    val testWords = tokenizer.transform(test)
    val testFeatures = hashingTF.transform(testWords)
    val testIdf = idfModel.transform(testFeatures)
    val predictions = model.transform(testIdf)
    //转化器，将索引进行反索引得到真实类别
    val converter = new IndexToString().setLabels(Array("0", "8", "4", "9", "5", "6", "1", "2", "7", "3")).setInputCol("prediction").setOutputCol("category") //traing.schema(indexer.getOutputCol).metadata
    val category = converter.transform(predictions).select("category")
    category.show

  }
}
