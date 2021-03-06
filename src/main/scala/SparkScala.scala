import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.regression.LabeledPoint

import scala.util.Random

object SparkScala {
  val NUMBER_FEATURES = 2000
  val TOPNWORDS = 0

  def main(args: Array[String]) {

    // PIPELINE START

    //Spark Context
    val conf:SparkConf = new SparkConf().setAppName("SpamClassifier").setMaster("local")
    val sc:SparkContext = new SparkContext(conf)

    //Load Data
    val raw_spam_training = sc.textFile("src/main/resources/spam_training.txt")
    val raw_nospam_training = sc.textFile("src/main/resources/nospam_training.txt")
      //FOR TEST, TO NOT USE
    val raw_spam_testing = sc.textFile("src/main/resources/spam_testing.txt")
    val raw_nospam_testing = sc.textFile("src/main/resources/nospam_testing.txt")

    //Data preprocessing
    val spam_training = preProcessData(raw_spam_training)
    val nospam_training = preProcessData(raw_nospam_training)
    val spam_testing = preProcessData(raw_spam_testing)
    val nospam_testing = preProcessData(raw_nospam_testing)

    val hashingTF = new HashingTF(NUMBER_FEATURES)
    //TRAINING: Feature representation and weighting
    val spamFeatures = spam_training.map(line => hashingTF.transform(line.split(" ")))
    val noSpamFeatures = nospam_training.map(line => hashingTF.transform(line.split(" ")))
    //TESTING: Feature representation and weighting
    val spamTESTFeatures = spam_testing.map(line => hashingTF.transform(line.split(" ")))
    val noSpamTESTFeatures = nospam_testing.map(line => hashingTF.transform(line.split(" ")))

    //TRAINING: Feature labelling
    val positiveLabels = spamFeatures.map(features => LabeledPoint(1, features))
    val negativeLabels = noSpamFeatures.map(features => LabeledPoint(0, features))
    //TESTING: Feature labelling
    val positiveTESTLabels = spamTESTFeatures.map(features => LabeledPoint(1, features))
    val negativeTESTLabels = noSpamTESTFeatures.map(features => LabeledPoint(0, features))

    //Merge positive and negative training samples
    val training_data = positiveLabels.union(negativeLabels)
    training_data.mapPartitions(Random.shuffle(_))
    training_data.cache()
    //Merge positive and negative test samples
    val test_data = positiveTESTLabels.union(negativeTESTLabels)
    test_data.mapPartitions(Random.shuffle(_))
    test_data.cache()

    //Train Naive Bayes model
    val model = NaiveBayes.train(training_data, 1.0)

    //Predictions on Test Data
    val predictionLabel = test_data.map(x => (model.predict(x.features), x.label))
    //Accuracy calculation on Test Data
    val accuracy = evaluate(predictionLabel, test_data)

    println("Naive Bayes Accuracy: "+accuracy)

    val spamCustomFeatures = hashingTF.transform("Dear, you have won $1,000,000!!! Send me your bank details ASAP".split(" "))
    val noSpamCustomFeatures = hashingTF.transform("Hey, this is Hichame. I love running and taking pictures.".split(" "))

    println("Spam case: "+model.predict(spamCustomFeatures))
    println("No spam case: "+model.predict(noSpamCustomFeatures))
  }

def evaluate(label : RDD[(Double, Double)] , data : RDD[_]) : Double = {
   return 1.0 * label.filter{case (x,y) => x == y}.count()/data.count()
}

  def preProcessData(Data: RDD[String]): RDD[String] = {
    /*
      PREPROCESSING THE DATA IN THIS CASE, LEADS TO A LOWER ACCURACY, SO I WILL SKIP IT.
      Naive Bayes: 0.9658886894075404 : 2000 Features + Lower / Number Preprocess
      Naive Bayes: 0.9676840215439856 : 2000 Features + Number Preprocess
      Naive Bayes: 0.8599640933572711 : 2000 Features + Lower / Number / Strip Preprocess
      Naive Bayes: 0.8662477558348295 : 10 Features + Lower / Number / Strip Preprocess
     */

    //Lowercase Data
    //val step = Data.map( x => x.toLowerCase())

    //Normalize numbers
    //val sstep = step.map( x => x.replaceAll("[0-9]+", "XNUMBERX "))

    //Remove all chars apart from regex
    //val ssstep = sstep.map( x => x.replaceAll("[^a-zA-Z0-9]", ""))

    /*Removing TopN most frequent words, lowers the accuracy
    val sssstep = Data
      .flatMap(line => line.split(" "))
      .map( word => (word, 1))
      .reduceByKey((a,b) => a + b)
      .map(item => item.swap)
      .sortByKey(false)
      .top(TOPNWORDS)

    val ssssstep = Data
      .flatMap(line => line.split(" "))
      .filter(x => !(sssstep.mkString(" ") contains x))
      .collect{ case i => i }
    */

    return Data//.map( x => x.replaceAll("""[\p{Punct}]""", ""))
  }
}