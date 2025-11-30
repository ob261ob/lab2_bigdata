from pyspark import SparkContext
sc = SparkContext("local[1]", "Test")

rdd = sc.parallelize(["тест слово ещё одно"])
rdd2 = rdd.flatMap(lambda line: line.split())
print(rdd2.collect())

sc.stop()
