# EXPLORING CHANGES TO THE CITY OF SEATTLE ----

# Evaluate the claim that “The City of Seattle is changing” 
# (in large part due to the growing technology industry) 
# by analyzing construction projects as documented 
# through building permit data14 downloaded 
# from the City of Seattle’s open data program.

install.packages("plotly") # once per machine 
library("plotly") # in each relevant script

install.packages("leaflet") # once per machine 
library("leaflet") # in each relevant script

install.packages("ggplot2") # once per machine 
library("ggplot2") # in each relevant script

# Load data ----
# downloaded from https://data.seattle.gov/Permitting/Building-Permits/76t5-zqzr
all_permits <- read.csv("data/r/Building_Permits.csv", stringsAsFactors = FALSE)

# Filter for permits for new buildings issued in 2010 or later
new_buildings <- all_permits %>% 
  filter(
    PermitTypeDesc == "New",
    PermitClass != "N/A",
    as.Date(all_permits$IssuedDate) >= as.Date("2010-01-01") # filter by date
)

View(new_buildings)

# Data wrangling ----

# Before mapping these points, you may want to get a higher-level view of the data. 
# For example, you could aggregate the data to show the number of permits issued per year. 
# This will again involve a bit of data wrangling, which is often 
# the most time-consuming part of visualization

# Create a new column storing the year the permit was issued
new_buildings <- new_buildings %>%
  mutate(year = substr(IssuedDate, 1, 4)) # extract the year

# Calculate the number of permits issued by year
by_year <- new_buildings %>% 
  group_by(year) %>% 
  count()

# Use plotly to create an interactive visualization of the data
plot_ly(
  data = by_year, # data frame to show
  x = ~year, # variable for the x-axis, specified as a formula
  y = ~n, # variable for the y-axis, specified as a formula
  type = "bar", # create a chart of type "bar" - a bar chart
  alpha = .7, # adjust the opacity of the bars
  hovertext = "y" # show the y-value when hovering over a bar
) %>% layout(
  title = "Number of new building permits per year in Seattle", 
  xaxis = list(title = "Year"),
  yaxis = list(title = "Number of Permits")
)

# After understanding this high-level view of the data, 
# you likely want to know where buildings are being constructed. 

# Create a Leaflet map, adding map tiles and circle markers ----
leaflet(data = new_buildings) %>% 
  addProviderTiles("CartoDB.Positron") %>%
  setView(lng = -122.3321, lat = 47.6062, zoom = 10) %>% 
  addCircles(
    lat = ~Latitude, # specify the column for `lat` as a formula
    lng = ~Longitude, # specify the column for `lng` as a formula
    stroke = FALSE, # remove border from each circle
    popup = ~Description, # show the description in a popup
  )

# you could use information about the permit classification 
# (i.e., if the permit is for a home versus a commercial building) 
# to color the individual circles. 
 
# Construct a function that returns a color based on the PermitClass colum 
# Colors are taken from the ColorBrewer Set3 palette
palette_fn <- colorFactor(palette = "Set3", domain = new_buildings$PermitClass)

# Modify the `addCircles()` method to specify color using `palette_fn()`
leaflet(data = new_buildings) %>% 
  addProviderTiles("CartoDB.Positron") %>%
  setView(lng = -122.3321, lat = 47.6062, zoom = 10) %>% 
  addCircles(
    lat = ~Latitude, # specify the column for `lat` as a formula
    lng = ~Longitude, # specify the column for `lng` as a formula
    stroke = FALSE, # remove border from each circle
    popup = ~Description, # show the description in a popup
    color = ~palette_fn(PermitClass)
  )

# Create a Leaflet map of new building construction by category
leaflet(data = new_buildings) %>% 
  addProviderTiles("CartoDB.Positron") %>%
  setView(lng = -122.3321, lat = 47.6062, zoom = 10) %>% 
  addCircles(
    lat = ~Latitude, # specify the column for `lat` as a formula
    lng = ~Longitude, # specify the column for `lng` as a formula
    stroke = FALSE, # remove border from each circle
    popup = ~Description, # show the description in a popup
    color = ~palette_fn(PermitClass)
  ) %>%
  # Add a legend layer in the "bottomright" of the map
  addLegend(
    position = "bottomright",
    title = "New Buildings in Seattle",
    pal = palette_fn, # the olor palette described by the legend 
    values = ~PermitClass, # the data values described by the legend 
    opacity = 1
  )

