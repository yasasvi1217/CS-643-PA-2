from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .getOrCreate()

# Load training and validation data
training = spark.read.format("csv").option("header","true").option("inferSchema", "true").load("TrainingDataset.csv")
validation = spark.read.format("csv").option("header","true").option("inferSchema", "true").load("ValidationDataset.csv")

# Define a VectorAssembler to combine all features into one vector
features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", 
                "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]  # Replace with your actual feature column names
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Define the Logistic Regression model
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Define a pipeline that chains the assembler and model
pipeline = Pipeline(stages=[assembler, lr])

# Fit the model to the training data
model = pipeline.fit(training)

# Make predictions on the validation data
predictions = model.transform(validation)

# Evaluate the model's performance
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

# Save the trained model
model.save("/home/ubuntu/model")
