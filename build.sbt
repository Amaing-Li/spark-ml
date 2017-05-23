name := "spark-ml"

version := "1.0"

scalaVersion := "2.12.2"


// https://mvnrepository.com/artifact/org.apache.hbase/hbase-client
libraryDependencies += "org.apache.hbase" % "hbase-client" % "1.2.4"

// https://mvnrepository.com/artifact/org.apache.hbase/hbase
libraryDependencies += "org.apache.hbase" % "hbase" % "1.2.4"

// https://mvnrepository.com/artifact/org.apache.hbase/hbase-common
libraryDependencies += "org.apache.hbase" % "hbase-common" % "1.2.4"

// https://mvnrepository.com/artifact/org.apache.hbase/hbase-server
libraryDependencies += "org.apache.hbase" % "hbase-server" % "1.2.4"

// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-hdfs
libraryDependencies += "org.apache.hadoop" % "hadoop-hdfs" % "2.7.3"

// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-common
libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "2.7.3"


libraryDependencies +=  "mysql" % "mysql-connector-java" % "5.1.18"

// https://mvnrepository.com/artifact/org.apache.phoenix/phoenix-core
libraryDependencies += "org.apache.phoenix" % "phoenix-core" % "4.10.0-HBase-1.2"

// https://mvnrepository.com/artifact/org.apache.phoenix/phoenix
libraryDependencies += "org.apache.phoenix" % "phoenix" % "4.10.0-HBase-1.2"

// https://mvnrepository.com/artifact/org.apache.spark/spark-mllib_2.11
libraryDependencies += "org.apache.spark" % "spark-mllib_2.11" % "2.1.0"

// https://mvnrepository.com/artifact/com.databricks/spark-csv_2.10
libraryDependencies += "com.databricks" % "spark-csv_2.10" % "1.5.0"
