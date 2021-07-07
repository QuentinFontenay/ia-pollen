from pyspark.ml import Pipeline
# from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer, StandardScaler, VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.types import IntegerType
from fbprophet import Prophet
import matplotlib as mpl
from fbprophet.plot import plot_plotly
import pandas as pd


conf = SparkConf()
conf.setMaster('spark://172.21.0.3:7077')
conf.setAppName('IA pollen')

sc = SparkContext(conf=conf)
spark = SQLContext(sc)
df = spark.read.format("csv").option('header', 'true').load("cassandra.csv")

df_sort = df.orderBy(['Date'], ascending = [True])

df_sort.show(5)


my_col = df_sort.select(['Date','MaxAirTempC','MinAirTempC','PrecipitationC', 'Alnus'])
data_df = my_col.withColumn("Alnus", my_col["Alnus"].cast(IntegerType()))
data_df.show(5)

pandas = data_df.toPandas()
train_dataset= pd.DataFrame()
train_dataset['ds'] = pd.to_datetime(pandas["Date"])
train_dataset['y'] = pandas['Alnus']

prophet_model = Prophet()
prophet_model.fit(train_dataset)
period_in_hours = 24 * 30
future = prophet_model.make_future_dataframe(periods=period_in_hours, freq='H')
forecast = prophet_model.predict(df=future)
print(forecast)
fig1=prophet_model.plot(forecast)

fig1.show()
# MaxAirTempC_indexer = StringIndexer(inputCol = 'MaxAirTempC', outputCol = 'MaxAirTempCIndex')
# MaxAirTempC_encoder = OneHotEncoder(inputCol='MaxAirTempCIndex', outputCol = 'MaxAirTempCVec')

# MinAirTempC_indexer = StringIndexer(inputCol = 'MinAirTempC', outputCol = 'MinAirTempCIndex')
# MinAirTempC_encoder = OneHotEncoder(inputCol='MinAirTempCIndex', outputCol = 'MinAirTempCVec')

# PrecipitationC_indexer = StringIndexer(inputCol = 'PrecipitationC', outputCol = 'PrecipitationCIndex')
# PrecipitationC_encoder = OneHotEncoder(inputCol='PrecipitationCIndex', outputCol = 'PrecipitationCVec')

# assembler = VectorAssembler(inputCols = ['MaxAirTempCVec', 'MinAirTempCVec', 'PrecipitationCVec'], outputCol = 'features')

# log_reg = LogisticRegression(featuresCol = 'features', labelCol = 'Alnus')
# pipeline = Pipeline(stages = [MaxAirTempC_indexer, MinAirTempC_indexer, PrecipitationC_indexer,
#                              MaxAirTempC_encoder, MinAirTempC_encoder, PrecipitationC_encoder,
#                              assembler, log_reg])

# train, test = data_df.randomSplit([0.7, 0.3])
# fit_model = pipeline.fit(train)
# results = fit_model.transform(test)
# results.select('prediction', 'Alnus').show(3)