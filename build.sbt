name := "SparkScala"

version := "1.0"

scalaVersion := "2.12.8"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies ++= Seq("org.apache.spark" %% "spark-core" % "2.4.0",
                            "org.apache.logging.log4j" % "log4j-core" % "2.11.1",
                            "org.apache.spark" %% "spark-mllib" % "2.4.0",
                            "org.scalaj" %% "scalaj-http" % "2.4.1")