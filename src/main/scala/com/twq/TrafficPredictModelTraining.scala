package com.twq

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StringType}

object TrafficPredictModelTraining {
  def main(args: Array[String]): Unit = {
    // 1. Spark应用的配置
    val conf = new SparkConf()
    if (!conf.contains("spark.master")) {
      conf.setMaster("local")
    }

    val spark = SparkSession
      .builder()
      .appName("TrafficPredictModelTraining")
      .config(conf)
      .getOrCreate()

    // 2. 加载数据
    var trafficDF = spark.read
      .option("header", "true")
      .csv("data/traffic.csv")
      .select(col("time").cast(StringType),
        col("state").cast(DoubleType),
        col("congestionDistance").cast(DoubleType),
        col("speed").cast(DoubleType),
        col("temperature").cast(DoubleType),
        col("visibility").cast(DoubleType),
        col("windSpeed").cast(DoubleType),
        col("skycon").cast(StringType))

    // 3. 数据转换

    // 3.1 将 time 转成long类型的timestamp
    // yyyy-MM-dd HH:mm => timestamp , 比如：2020-03-01 20 => long
    trafficDF = trafficDF.withColumn("time", unix_timestamp(concat(col("time"), lit(":00")), "yyyy-MM-dd HH:mm"))


    val minTimestamp = unix_timestamp(lit("2020-03-01"), "yyyy-MM-dd")
    trafficDF = trafficDF.withColumn("time", col("time") - minTimestamp)

    // 3.2 将部分String类型转成Double类型，因为Spark在模型训练的时候只认类型为Double的特征值
    trafficDF = trafficDF.withColumn("time", col("time").cast(DoubleType))

    // 离散值或者说类别值
    // Spark实现机器学习的模型训练中，不认字符串类型的类别值
    // 3.3 我们要把字符串类型的类别值转成数值类型的类别值
    val stringIndexer = new StringIndexer().setInputCol("skycon").setOutputCol("skyconIndex")
    trafficDF = stringIndexer.fit(trafficDF).transform(trafficDF)


    // 3.4 删除不用的列
    trafficDF = trafficDF.drop("skycon")

    // 3.5：OneHot编码
    // 线性回归对输入为离散值的值有特殊要求，也就是说需要经过转换成指定的格式才可以
    // state：1，2，3，4，5，6，7, 8, 9, 10
    // 1 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    // 2 => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    // 3 => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    // 4 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    // 5 => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    // 6 => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    // 7 => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    val oneHotEncoderEstimator = new OneHotEncoderEstimator()
      .setInputCols(Array("state", "skyconIndex"))
      .setOutputCols(Array("dummyState", "dummySkyConIndex"))
    trafficDF = oneHotEncoderEstimator.fit(trafficDF).transform(trafficDF)

    // 删除非onehot编码的列
    trafficDF = trafficDF.drop("state").drop("skyconIndex")

    // 4、构建线性回归模型并且预测

    // 4.1 将数据集切分成训练数据集和测试数据集
    var Array(training, testing) = trafficDF.randomSplit(Array(0.8, 0.2))

    // 4.2 线性回归的输入是向量
    // 需要将DF中的所有的列构建成一个特征向量
    val labelColumn = "congestionDistance"
    val vectorAssembler = new VectorAssembler()
      .setInputCols(training.drop(labelColumn).columns)
      .setOutputCol("features")

    training = vectorAssembler.transform(training)
    training.select("features").show(truncate = false)
    /*testing = vectorAssembler.transform(testing)

    // 4.3 构建线性回归模型
    val lr = new LinearRegression()
      .setFeaturesCol("features") // 输入的特征矩阵
      .setLabelCol(labelColumn) // 要预测的值
      .setMaxIter(5)
      .setRegParam(1.0)
      //.setElasticNetParam(0)

    //val model = lr.fit(training)

    val pipeline = new Pipeline().setStages(Array(vectorAssembler, lr))

    //val model: PipelineModel = pipeline.fit(training)

    //val testDF = model.transform(testing)

    val regressionEvaluator = new RegressionEvaluator()
      .setLabelCol(labelColumn)
      .setPredictionCol("prediction")
      .setMetricName("rmse") // 均方根误差

    //println("rmse = " + regressionEvaluator.evaluate(testDF))

    val maxIters = Array(5, 20, 100)
    val regParams = Array(0.1, 1.0, 10.0)

    // 3 * 3 = 9
    val paramMap = new ParamGridBuilder()
      .addGrid(lr.getParam("maxIter"), maxIters)
      .addGrid(lr.getParam("regParam"), regParams)
      .build

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(regressionEvaluator)
      .setNumFolds(5)
      .setEstimatorParamMaps(paramMap)

    val model = crossValidator.fit(training)

    // 最佳模型
    val bestModel = model.bestModel.asInstanceOf[PipelineModel]
    val lrModel = bestModel.stages(1).asInstanceOf[LinearRegressionModel]

    val testDF = lrModel.transform(testing)
    println("rmse = " + regressionEvaluator.evaluate(testDF))

    // 保存模型
    lrModel.write.save("hdfs://localhost:9000/toby/test")

    spark.stop()*/
  }
}
