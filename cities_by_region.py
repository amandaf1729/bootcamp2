# API: https://www.metaweather.com/api/
import urllib
import requests
import time

US_woeid = "23424977"
url = "https://www.metaweather.com/api/location/"
US_url = url + US_woeid
data = requests.get(US_url).json()

 # Split and store all the states by region
new_england = ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "Rhode Island", "Vermont"]
middle_atlantic = ["Delaware", "Maryland", "New Jersey", "New York", "Pennsylvania"]
south = ["Alabama", "Arkansas", "Florida", "Georgia", "Kentucky", "Louisiana", "Mississippi", "Missouri", "North Carolina", "South Carolina", "Tennessee", "Virginia", "West Virginia"]
midwest = ["Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Nebraska", "North Dakota", "Ohio"," South Dakota", "Wisconsin"]
south_west = ["Arizona","New Mexico", "Oklahoma", "Texas"]
west = ["Alaska", "California", "Colorado", "Hawaii", "Idaho", "Montana", "Nevada", "Oregon", "Utah", "Washington", "Wyoming"]

regions = {"New England" : new_england, "Middle Atlantic" : middle_atlantic, "South" : south, "Midwest" : midwest, "South West": south_west, "West" : west}

# Store each of the cities and their woeid provided by the API by region
cities_by_region = {"New England" : [], "Middle Atlantic" : [], "South" : [], "Midwest" : [], "South West": [], "West" : []}
for state in data.get("children"):
    state_woeid = str(state.get("woeid"))
    states = requests.get(url + state_woeid).json()
    state_title = state.get("title")
    for region, states_list in regions.items():
        if state_title in states_list:
            cities = states.get("children")
            for city in cities:
                cities_by_region.get(region).append({"city": city.get("title") , "woeid" : city.get("woeid")})
print (cities_by_region)

# We stored the resulting cities_by_region dict so that we don't have to call the API every single time since it is very slow
c_by_r = {'New England': [{'city': 'Bridgeport', 'woeid': 2368947}, {'city': 'Portland', 'woeid': 2475688}, {'city': 'Boston', 'woeid': 2367105}, {'city': 'Manchester', 'woeid': 2444674}, {'city': 'Providence', 'woeid': 2477058}, {'city': 'Burlington', 'woeid': 2372071}], 'Middle Atlantic': [{'city': 'Wilmington', 'woeid': 2521358}, {'city': 'Baltimore', 'woeid': 2358820}, {'city': 'Newark', 'woeid': 2459269}, {'city': 'New York', 'woeid': 2459115}, {'city': 'Philadelphia', 'woeid': 2471217}], 'South': [{'city': 'Birmingham', 'woeid': 2364559}, {'city': 'Little Rock', 'woeid': 2440351}, {'city': 'Jacksonville', 'woeid': 2428344}, {'city': 'Miami', 'woeid': 2450022}, {'city': 'Atlanta', 'woeid': 2357024}, {'city': 'Louisville', 'woeid': 2442327}, {'city': 'New Orleans', 'woeid': 2458833}, {'city': 'Jackson', 'woeid': 2428184}, {'city': 'Kansas City', 'woeid': 2430683}, {'city': 'St. Louis', 'woeid': 2486982}, {'city': 'Charlotte', 'woeid': 2378426}, {'city': 'Raleigh', 'woeid': 2478307}, {'city': 'Columbia', 'woeid': 2383552}, {'city': 'Memphis', 'woeid': 2449323}, {'city': 'Nashville', 'woeid': 2457170}, {'city': 'Richmond', 'woeid': 2480894}, {'city': 'Virginia Beach', 'woeid': 2512636}, {'city': 'Charleston', 'woeid': 2378319}], 'Midwest': [{'city': 'Chicago', 'woeid': 2379574}, {'city': 'Indianapolis', 'woeid': 2427032}, {'city': 'Des Moines', 'woeid': 2391446}, {'city': 'Wichita', 'woeid': 2520077}, {'city': 'Detroit', 'woeid': 2391585}, {'city': 'Minneapolis', 'woeid': 2452078}, {'city': 'Omaha', 'woeid': 2465512}, {'city': 'Fargo', 'woeid': 2402292}, {'city': 'Columbus', 'woeid': 2383660}, {'city': 'Milwaukee', 'woeid': 2451822}], 'South West': [{'city': 'Mesa', 'woeid': 2449808}, {'city': 'Phoenix', 'woeid': 2471390}, {'city': 'Tucson', 'woeid': 2508428}, {'city': 'Albuquerque', 'woeid': 2352824}, {'city': 'Santa Fe', 'woeid': 2488867}, {'city': 'Oklahoma City', 'woeid': 2464592}, {'city': 'Austin', 'woeid': 2357536}, {'city': 'Dallas', 'woeid': 2388929}, {'city': 'El Paso', 'woeid': 2397816}, {'city': 'Fort Worth', 'woeid': 2406080}, {'city': 'Houston', 'woeid': 2424766}, {'city': 'San Antonio', 'woeid': 2487796}], 'West': [{'city': 'Anchorage', 'woeid': 2354490}, {'city': 'Bakersfield', 'woeid': 2358492}, {'city': 'Fresno', 'woeid': 2407517}, {'city': 'Lake Tahoe', 'woeid': 23511744}, {'city': 'Long Beach', 'woeid': 2441472}, {'city': 'Los Angeles', 'woeid': 2442047}, {'city': 'Mountain View', 'woeid': 2455920}, {'city': 'Oakland', 'woeid': 2463583}, {'city': 'Palm Springs', 'woeid': 2467696}, {'city': 'Sacramento', 'woeid': 2486340}, {'city': 'San Diego', 'woeid': 2487889}, {'city': 'San Francisco', 'woeid': 2487956}, {'city': 'San Jose', 'woeid': 2488042}, {'city': 'Santa Cruz', 'woeid': 2488853}, {'city': 'Boulder', 'woeid': 2367231}, {'city': 'Colorado Springs', 'woeid': 2383489}, {'city': 'Denver', 'woeid': 2391279}, {'city': 'Honolulu', 'woeid': 2423945}, {'city': 'Boise', 'woeid': 2366355}, {'city': 'Billings', 'woeid': 2364254}, {'city': 'Las Vegas', 'woeid': 2436704}, {'city': 'Portland', 'woeid': 2475687}, {'city': 'Salt Lake City', 'woeid': 2487610}, {'city': 'Seattle', 'woeid': 2490383}, {'city': 'Cheyenne', 'woeid': 2379552}]}