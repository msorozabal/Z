from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import mean as mean_func

from pyspark.ml import Pipeline

import concurrent.futures
""" concurrent.futures ThreadPoolExecutor nos permite ejecutar funciones en paralelo en distintos hilos o procesos. """

# Inicializar la sesión de Spark
spark = SparkSession.builder.appName("DataPipeline").getOrCreate()

# Cargar los datos en un DataFrame
df = spark.read.csv("data/backend-dev-data-dataset.txt", header=True, inferSchema=True)

#### FUNCIONES PASO 1: 

def check_schema(df):
    print("Data schema:")
    df.printSchema()    

def missing_data(df):
    for c in df.columns:
        df = df.withColumn(c, \
                      when(col(c)=="na", None) \
                      .otherwise(col(c))) 

    missing_data_count = [df.filter(col(c).isNull()).count() for c in df.columns]
    print("Number of rows with missing data per column: ", missing_data_count)
    
    # Acá me quedo duda con la letra, ya que dice "VERIFICAR" no pide corregirlo.
    # Si hubiera que corregirlo utilizaría una estrategia de Mediana o usar percentil 50 y retornaría el df.
    
def find_outliers(df, interval):
    for c in df.columns:
        col_type = df.schema[c].dataType

        if col_type == DoubleType():   
            mean = df.select(mean_func(col(c))).first()[0]
            std = df.select(stddev(col(c))).first()[0]
            
            # Se propone un estudio de outliers, pueden hacerse otros como box-plots, etc. 
            # Filtro las filas que tengan valor fuera de 3 standard deviations de la media
            outliers = df.filter((col(c) < (mean - interval * std)) | (col(c) > (mean + interval * std)))
            print(f'Total outliers for column {c} : {outliers.count()}')
                 
#### FUNCIONES PASO 2: 

def normalize_column(df, column):
    col_mean = df.agg(mean(f"{column}")).first()[0]
    col_std = df.agg(stddev(f"{column}")).first()[0]
    scaled_df = df.select("*", ((col(f"{column}") - col_mean) / col_std).alias(f"normalized_{column}"))
    return scaled_df
    
def filter_column(df, column, filter_value):
    df = df.filter(col(column) == filter_value)
    return df

def grouped_avg_data_monthly(df, group_column, date_column):
    grouped_df = df.groupBy(month(col(date_column)).alias("month"))
    grouped_df = grouped_df.agg(avg(group_column).alias(f"avg_{group_column}"))
    return grouped_df


#### FUNCIONES PASO 3: 

def transformed_column(df,x,y):
    transformed_column = pow(df[x], 3) + exp(df[y])
    df_transformed = df.withColumn(f"transformed_value_{x}_{y}", transformed_column)
    return df_transformed

def unique(df, column):
    unique_count = df.agg(countDistinct(col(f"{column}")).alias(f"unique_{column}_count"))
    return unique_count

#### KINESIS STREAM CONNECTOR: 

"""from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Step 1: Create a SparkSession
spark = SparkSession.builder.appName("StreamingInference").getOrCreate()

# Step 2: Read the data in real-time
df = spark.readStream \
    .format("kinesis") \
    .option("streamName", "kinesis_stream_name") \
    .option("awsAccessKey", "your_aws_access_key") \
    .option("awsSecretKey", "your_aws_secret_key") \
    .option("region", "us-west-2") \
    .load()

# Step 3: Transform the data
df_transformed = df.selectExpr("cast (data as string) as data") \
    .groupBy(window(df.timestamp, "10 minutes", "5 minutes")) \
    .agg(count("data").alias("count"))

# Step 4: Store the results
query = df_transformed.writeStream \
    .outputMode("complete") \
    .format("parquet") \
    .option("path", "/tmp/streaming_data") \
    .option("checkpointLocation", "/tmp/streaming_checkpoints") \
    .start()

# Step 5: Start the streaming process
query.awaitTermination()"""


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor() as executor:
    results = [executor.submit(check_schema, df), 
               executor.submit(missing_data, df),
               executor.submit(find_outliers, df, 3)]

    scaled_df = normalize_column(df, "cont_9")
    filtered_df = filter_column(df, "cat_7", "frequent")
    grouped_avg_monthly_df = grouped_avg_data_monthly(df,"cont_3", "date_2")


    filtered_df.show(10)
    grouped_avg_monthly_df.show(10)
    scaled_df.show(10)   

    df_transformed = transformed_column(df, "cont_4", "cont_9")
    df_unique = unique(df, "cat_8")

    df_transformed.show(10)
    df_unique.show(10)