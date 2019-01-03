import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector

object SparkScala {
  val NUMBER_FEATURES = 100

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
  }
}