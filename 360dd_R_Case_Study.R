# MAPPING EVICTIONS IN SAN FRANCISCO ----

# Before mapping this data, a minor amount of formatting needs to be done on the raw data set

# Load and format eviction notices data ----
# Data downloaded from https://catalog.data.gov/dataset/eviction-notices
# Load packages for data wrangling and visualization
library("dplyr") 
library("tidyr") 
library("ggmap")

# Load .csv file of notices
eviction_notices <- read.csv("data/r/Eviction_Notices.csv", stringsAsFactors = F)

View(eviction_notices)

# Data wrangling: format dates, filter to 2017 notices, extract lat/long data ----
notices <- eviction_notices %>%
  mutate(date = as.Date(File.Date, format="%m/%d/%y")) %>% 
  filter(format(date, "%Y") == "2017") %>%
  separate(Location, c("lat", "long"), ", ") %>% # split the column at the comma
  mutate(
    lat = as.numeric(gsub("\\(", "", lat)), # remove starting parentheses
    long = as.numeric(gsub("\\)", "", long)) # remove closing parentheses
  )

# Create a map of San Francisco, with a point at each eviction notice address 
# Use `install_github()` to install the version of `ggmap` on GitHub (ofte 
# devtools::install_github("dkhale/ggmap") # once per machine 
library("ggmap")
library("ggplot2")

# Create the background of map tiles ----
base_plot <- qmplot( 
  data = notices, # name of the data frame
  x = long, # data feature for longitude
  y = lat, # data feature for latitude
  geom = "blank", # don't display data points (yet) 
  maptype = "toner-background", # map tiles to query
  darken = .7, # darken the map tiles
  legend = "topleft" # location of legend on page
)

# You can store a plot returned by the ggplot() function in a variable (as in the preceding code)!
# This allows you to add different layers on top of a base plot, 
# or to render the plot at chosen locations throughout a report

# Add the locations of evictions to the map ----
base_plot +
  geom_point(mapping = aes(x = long, y = lat), color = "red", alpha = .3) +
  labs(title = "Evictions in San Francisco, 2017") +
  theme(plot.margin = margin(.3, 0, 0, 0, "cm")) # adjust spacing around the map

# Draw a heatmap of eviction rates, using ggplot2 to compute the shape/col ----
base_plot + 
  geom_polygon(
    stat = "density2d", # calculate the two-dimensional density of points 
    mapping = aes(fill = stat(level)), # use the computed density to set the fill 
    alpha = .3, # Set the alpha (transparency)
  ) +
  scale_fill_gradient2(
    "# of Evictions", 
    low = "white", 
    mid = "yellow", 
    high = "red" 
  ) +
  labs(title="Number of Evictions in San Francisco, 2017") + 
  theme(plot.margin = margin(.3, 0, 0, 0, "cm"))




