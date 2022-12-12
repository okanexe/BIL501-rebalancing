import pyspark
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.sql import functions as F
from  pyspark.sql import functions
from pyspark.sql.functions import col, radians, asin, sin, sqrt, cos
from pyspark.ml.regression import LinearRegression
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler


def timestampClearer(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name])
    df["hour"] = df.hour
    df["day_of_month"] = df.day
    df["day_of_week"] = df.dayofweek
    df["month"] = df.month
    return df


def read_datasets():
    citibike_csv_files = []
    for month in range(2, 13):
        file = "dataset/2021" + str(month) + "-citibike-tripdata.csv"
        citibike_csv_files.append(file)
    list_df = []
    for csv_file in citibike_csv_files:
        df = pd.read_csv(csv_file, index_col=None, header=0)
        list_df.append(df)
    citibike_df = pd.concat(list_df, axis=0, ignore_index=True)
    citibike_df.to_csv("dataset/citibike_df.csv", sep=",", encoding='utf-8', index=False)


def station_analysis(df):
    # times taking bike from start station
    df_start_station = df \
        .select('start_station_id', 'start_station_name', 'start_lng', 'start_lat') \
        .groupBy('start_station_id', 'start_station_name', 'start_lng', 'start_lat') \
        .agg(F.count('start_station_id').alias('n_ride')) \
        .toPandas()
    # times dropping bike end station
    df_end_station = df \
        .select('end_station_id', 'end_station_name', 'end_lng', 'end_lat') \
        .groupBy('end_station_id', 'end_station_name', 'end_lng', 'end_lat') \
        .agg(F.count('end_station_id').alias('n_ride')) \
        .toPandas()

    # number of stations have been used
    nunique_start_station = len(list(df_start_station.start_station_id.unique()))
    nunique_end_station = len(list(df_end_station.end_station_id.unique()))
    nunique_station = len(set(list(df_start_station.start_station_id.unique()) + list(df_end_station.end_station_id.unique())))
    print(f'There were {nunique_start_station:,} active start stations')
    print(f'There were {nunique_end_station:,} active end stations')
    print(f'There was total {nunique_station:,} active stations.')
    return nunique_start_station, nunique_end_station


def per_day(df):
    df_ts_day = df.selectExpr('left(started_at, 10) as day') \
        .groupBy('day') \
        .agg(F.count('day').alias('n_ride')) \
        .orderBy('day') \
        .toPandas()
    print(f'There was average {df_ts_day.n_ride.mean():,.2f} rides per day')


def per_month(df, nunique_start_station, nunique_end_station):
    # ride per station per month
    df_ts_m_station = df.selectExpr('left(started_at, 7) as month') \
        .groupBy('month') \
        .agg(F.count('month').alias('n_ride')) \
        .withColumn('ride_p_start_station', F.col('n_ride') / F.lit(nunique_start_station)) \
        .withColumn('ride_p_end_station', F.col('n_ride') / F.lit(nunique_end_station)) \
        .withColumn('avg_ride_p_station', (F.col('ride_p_start_station') + F.col('ride_p_end_station')) / 2) \
        .orderBy('month') \
        .toPandas()
    print(f'Each station served {df_ts_m_station.avg_ride_p_station.mean():,.2f} rides per month')


def prepare_dataset_to_ML(data):
    data = data.dropna("any")
    bike_df_full = (data.withColumn('year_start_date', F.year(dt.started_at))
                    .withColumn('month_start_date', F.month(dt.started_at))
                    .withColumn('day_start_date', F.dayofweek(dt.started_at))
                    .withColumn('hour_start_date', F.hour(dt.started_at))
                    .withColumn('year_end_date', F.year(dt.ended_at))
                    .withColumn('month_end_date', F.month(dt.ended_at))
                    .withColumn('day_end_date', F.dayofweek(dt.ended_at))
                    .withColumn('hour_end_date', F.hour(dt.ended_at))
                    )
    bike_df_full = bike_df_full.withColumn("dlon", radians(col("end_lng")) - radians(col("start_lng"))) \
        .withColumn("dlat", radians(col("end_lat")) - radians(col("start_lat"))) \
        .withColumn("haversine_dist", asin(sqrt(
        sin(col("dlat") / 2) ** 2 + cos(radians(col("start_lat")))
        * cos(radians(col("end_lat"))) * sin(col("dlon") / 2) ** 2
    )
    ) * 2 * 3963 * 5280) \
        .drop("dlon", "dlat") \

    bike_df_full = bike_df_full.withColumn('started_at', F.to_timestamp(F.col('started_at'))) \
        .withColumn('ended_at', F.to_timestamp(F.col('ended_at'))) \
        .withColumn('DiffInMinutes', (F.col("ended_at").cast("long") - F.col('started_at').cast("long")) / 60)
    return bike_df_full


def lr(dt):
    bike_dt = prepare_dataset_to_ML(dt)
    bike_dt = bike_dt.drop("ride_id", "start_station_name", "end_station_name", "started_at", "ended_at")
    bike_dt.show(truncate=False)
    bike_dt = bike_dt.withColumn("start_station_id", F.col("start_station_id").cast(FloatType())). \
        withColumn("end_station_id", F.col("end_station_id").cast(FloatType())). \
        withColumn("start_lat", F.col("start_lat").cast(FloatType())). \
        withColumn("start_lng", F.col("start_lng").cast(FloatType())). \
        withColumn("end_lat", F.col("end_lat").cast(FloatType())). \
        withColumn("end_lng", F.col("end_lng").cast(FloatType()))
    vectorAssembler = VectorAssembler(
        inputCols=["start_station_id", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng",
                   "year_start_date", "year_start_date",
                   "day_start_date", "hour_start_date", "year_end_date",
                   "month_end_date", "day_end_date", "hour_end_date",
                   "DiffInMinutes"], outputCol='features')
    ml_df = vectorAssembler.setHandleInvalid("skip").transform(bike_dt)
    ml_df = ml_df.select(['features', 'haversine_dist'])
    ml_df.show(3)
    splits = ml_df.randomSplit([0.7, 0.3])
    train_df = splits[0]
    test_df = splits[1]

    lr = LinearRegression(featuresCol='features', labelCol='haversine_dist')
    lr_model = lr.fit(train_df)
    print("Coefficients: " + str(lr_model.coefficients))
    print("Intercept: " + str(lr_model.intercept))

    trainingSummary = lr_model.summary
    print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
    print("r2: %f" % trainingSummary.r2)


if __name__ == '__main__':
    # read_datasets()
    spark = SparkSession.builder.appName('scooter').getOrCreate()
    dt = spark.read.option("header", "true").csv("dataset/citibike_df_test.csv")
    dt = dt.drop("rideable_type", "member_casual")
    print(dt.count())
    dt.printSchema()
    print("raw dataset: ")
    dt.show(10)
    dt.describe().show()
    nunique_start_station, nunique_end_station = station_analysis(dt)
    per_day(dt)
    per_month(dt, nunique_start_station, nunique_end_station)
    # divide timestamp for year-month-hour ex.
    dt.withColumn("start_year", functions.year(dt.started_at)). \
        withColumn("start_month", functions.month(dt.started_at)). \
        withColumn("start_hour", functions.hour(dt.started_at)).select("started_at", "start_year", "start_month", "start_hour"). \
        show()
    # hourly riding ex.
    dt.withColumn("start_hour", functions.hour(dt.started_at)).groupBy("start_hour").agg(F.count("ride_id")).orderBy("start_hour").show()
    df_outflow = dt.groupBy(functions.date_trunc("hour", functions.col("started_at")), "start_station_id", "start_station_name").count()
    df_inflow = dt.groupBy(functions.date_trunc("hour", functions.col("ended_at")), "end_station_id", "end_station_name").count()
    # dfn.write.csv("dataset/ds")
    df_inflow.show()
    df_outflow.show()


    ### DATA ANALYSIS ###

    dt.show(1, vertical=True)
    sid = dt.where(dt.start_station_id.isNull()).count()
    print("start station id count=> ", sid)
    eid = dt.where(dt.end_station_id.isNull()).count()
    print("end station id count=> ", eid)
    dt_any = dt.dropna("any")
    print("before dropna: " , dt.count() , " after dropna: " , dt_any.count())
    print("deleted rows: ", dt.count()-dt_any.count())
    dt_any.groupBy("start_station_id", "start_station_name").count().orderBy(F.col("count").desc()).show()
    dt_any.groupBy("end_station_id", "end_station_name").count().orderBy(F.col("count").desc()).show()
    # Count total number of unique stations
    print("Count total number of unique stations: ", dt_any.select("start_station_id").union(dt_any.select("end_station_id")).distinct().count())

    df2 = dt.withColumn('started_at', F.to_timestamp(F.col('started_at'))) \
        .withColumn('ended_at', F.to_timestamp(F.col('ended_at'))) \
        .withColumn('DiffInSeconds', F.col("ended_at").cast("long") - F.col('started_at').cast("long"))
    df2.show()
    # Calculate average time of single rental(minute)
    print("Calculate average time of single rental(minute): ")
    df2.select((F.avg("DiffInSeconds")/60).alias("average")).show()
    # Find stations with the most traffic between them
    print("Find stations with the most traffic between them: ")
    df2.select(F.when(df2["start_station_id"] > df2["end_station_id"],
                         F.array(df2["start_station_id"], df2["end_station_id"])) \
                  .otherwise(F.array(df2["end_station_id"], df2["start_station_id"])) \
                  .alias("route")) \
        .groupBy("route") \
        .count() \
        .orderBy(F.desc("count")) \
        .show(1)
    # Show stations from route above
    print("Show stations from route above: ")
    df2.filter((df2.start_station_id == 5669.1) | (df2.start_station_id == 5626.13)) \
        .select("start_station_name").distinct().show(truncate=False)

    # Find rush hour
    print("Find rush hour: ")
    df2.select(F.hour("started_at").alias("hour")) \
        .groupBy("hour").count().orderBy(F.desc("count")).show(7)

    # Find rush day
    # from 1 for a Sunday through to 7 for a Saturday
    print("Find rush day: ")
    df2.select(F.dayofweek("started_at").alias("day")) \
        .groupBy("day").count().orderBy(F.desc("count")).show(7)

    # Find rush months
    print("Find rush months: ")
    df2.select(F.month("started_at").alias("month")) \
        .groupBy("month").count().orderBy(F.desc("count")).show(7)

    # Find average rentals grouped by weekday
    print("Find average rentals grouped by weekday")
    df2.select(F.date_format("started_at", "dd.MM.yyyy").alias("date"), \
                  F.date_format("started_at", "E").alias("weekday")) \
        .groupBy("date", "weekday").count() \
        .groupBy("weekday").agg(F.avg("count").alias("avg_use")) \
        .orderBy("avg_use").show()

    # find distance between start and end station as haversine
    print("find distance between start and end station as haversine: ")
    df_haversine = df2.withColumn("dlon", radians(col("end_lng")) - radians(col("start_lng"))) \
        .withColumn("dlat", radians(col("end_lat")) - radians(col("start_lat"))) \
        .withColumn("haversine_dist", asin(sqrt(
        sin(col("dlat") / 2) ** 2 + cos(radians(col("start_lat")))
        * cos(radians(col("end_lat"))) * sin(col("dlon") / 2) ** 2
    )
    ) * 2 * 3963 * 5280) \
        .drop("dlon", "dlat") \
        .show(truncate=False)

    # 'n_routes' : total number of unique trips combinations (x -> y == y -> x) of that day
    temp = df2.select(F.date_format("started_at", "dd.MM.yyyy").alias("date"),
                         F.when(df2["start_station_id"] > df2["end_station_id"],
                                F.array(df2["start_station_id"], df2["end_station_id"]))
                         .otherwise(F.array(df2["end_station_id"], df2["start_station_id"])).alias("route")) \
        .groupBy("date").agg(F.countDistinct("route").alias("n_routes")).show()


    ## hourly trip count and unique route count
    print("hourly trip count and unique route count: ")
    dailyData = dt.withColumn("date", F.date_format("started_at", "dd.MM.yyyy HH:00")) \
        .groupBy("date") \
        .agg(F.count("*").alias("n_trips"))
    temp = dt.select(F.date_format("started_at", "dd.MM.yyyy HH:00").alias("date"),
                         F.when(dt["start_station_id"] > dt["end_station_id"],
                                F.array(dt["start_station_id"], dt["end_station_id"])) \
                         .otherwise(F.array(dt["end_station_id"], dt["start_station_id"])).alias("route")) \
        .groupBy("date").agg(F.countDistinct("route").alias("n_routes"))
    dailyData.join(temp, "date").orderBy("date").show()