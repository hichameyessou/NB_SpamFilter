import org.apache.spark.ml.feature.IDF
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint

object SparkScala {
  val NUMBER_FEATURES = 2000

  def main(args: Array[String]) {

    //
    val conf:SparkConf = new SparkConf().setAppName("SpamClassifier").setMaster("local")
    val sc:SparkContext = new SparkContext(conf)

    // load data as spark-datasets
    val spam_training = sc.textFile("src/main/resources/spam_training.txt")
    val nospam_training = sc.textFile("src/main/resources/nospam_training.txt")

    // TESTING, TO NOT USE
    val spam_testing = sc.textFile("src/main/resources/spam_testing.txt")
    val nospam_testing = sc.textFile("src/main/resources/nospam_testing.txt")

    // implement: convert datasets to either rdds or dataframes (your choice) and build your pipeline
    val hashingTF = new HashingTF(NUMBER_FEATURES)
    val spamFeatures = spam_training.map(line => hashingTF.transform(line.split(" ")))
    val noSpamFeatures = nospam_training.map(line => hashingTF.transform(line.split(" ")))
    // FEATURES ON TEST, FOR TESTING
    val spamTESTFeatures = spam_testing.map(line => hashingTF.transform(line.split(" ")))
    val noSpamTESTFeatures = nospam_testing.map(line => hashingTF.transform(line.split(" ")))

    val positiveExamples = spamFeatures.map(features => LabeledPoint(1, features))
    val negativeExamples = noSpamFeatures.map(features => LabeledPoint(0, features))
    val training_data = positiveExamples.union(negativeExamples)
    training_data.cache()


    val positiveTESTExamples = spamTESTFeatures.map(features => LabeledPoint(1, features))
    val negativeTESTExamples = noSpamTESTFeatures.map(features => LabeledPoint(0, features))
    val test_data = positiveTESTExamples.union(negativeTESTExamples)
    test_data.cache()

    val model = NaiveBayes.train(training_data, 1.0)

    val predictionLabel = test_data.map(x => (model.predict(x.features), x.label))

    val accuracy = 1.0 * predictionLabel.filter{case (x,y) => x == y}.count()/test_data.count()

    predictionLabel.saveAsTextFile("src/main/resources")
    println("ACCURACY: "+accuracy)

  }
}