Citibike Dataset: https://s3.amazonaws.com/tripdata/index.html

When you download datasets directly, the filenames appear as 20120102. If you edit the date names as 201212 by removing zeros, the read_dataset function will work properly.

Then you can start analyzing the data using the csv file that will be created with the read_dataset function