HCS Bootcamp 2

Amanda Fernandez, Peter Bynum, Zuzanna Skoczylas

We chose to work with the MetaWeather API (https://www.metaweather.com/api/) because we thought it would be interesting to notice trends in different city's weather and try to predict what region a city is in based on its weather. In our cities_by_region.py file, we called the API to obtain the different US cities that it had data for. This API had some international cities as well so we queried the woeid for the United States and obtained all the cities that were in the US. 

Then, in our cities_data.py file, we iterated through all of the cities to call the API again, this time obtaining the weather information for each. We were interested in the temperature, humidity, and wind speed data points since we thought these would be good indicators for what region the city is in. Querying so many cities (76 in total) took a few minutes to run, so we printed the final weather_data dictionary and used that for our clustering.

In our clustering.py, we used the K Nearest Neighbors method by modifying the example code to fit our data. We found that the module was not as accurate as we expected, giving us approximately 47.83% accuracy. Overall, we saw some trends in a region's weather but it was not as defined as we were hoping for. 
