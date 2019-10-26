
# --------------------------
# Analyzing Flight Data ----
# --------------------------

# Load tidyverse ----
install.packages("tidyverse")
library("tidyverse")

# Load flights data frame ----
install.packages("nycflights13")
library("nycflights13")

# Getting to know the flights data set ----

# Read available documentation
?flights

# Check the number of rows and columns
dim(flights)

# Inspect the column names
colnames(flights)

# Look at the data frame in the RStudio Viewer
View(flights)

head(flights)

# Questions ----
# 1. Which airline has the highest number of delayed departures? ----

?n

has_most_delays <- flights %>%
  group_by(carrier) %>%
  filter(dep_delay > 0) %>%
  summarize(num_delay = n()) %>%
  filter(num_delay == max(num_delay)) %>%
  select(carrier)

print(as.data.frame(has_most_delays))

View(airlines)

# Get name of the most delayed carrier
most_delayed_name <- has_most_delays %>%
  left_join(airlines, by = 'carrier') %>%
  select(name)

print(most_delayed_name$name)

# We should check the proportion of delayed flights the have more fair answer
# as UA is a big airline

# number of all flights per carrier
all_flight_stats <- flights %>%
  group_by(carrier) %>%
  summarize(
    num_all_flights = n()
  )

# number of delayed flights per carrier
delay_flight_stats <- flights %>%
  group_by(carrier) %>%
  filter(dep_delay > 0) %>%
  summarize(
    num_delay = n()
  )

# combine per carrier summaries
flight_stats <- all_flight_stats %>%
  left_join(delay_flight_stats, by = "carrier")

# add ratio column
flight_stats <- flight_stats %>%
  mutate(
    ratio = flight_stats$num_delay / flight_stats$num_all_flights
  )

has_most_delays_by_ratio <- flight_stats %>%
  filter(ratio == max(ratio)) %>%
  select(carrier, ratio)

print(has_most_delays_by_ratio)

# Get name of the most delayed carrier
most_delayed_name_by_ratio <- has_most_delays_by_ratio %>%
  left_join(airlines, by = 'carrier') %>%
  select(name)

print(most_delayed_name_by_ratio)

# It seems, that Southwest Airlines Co. - WN - with more than 53% of delayed flights wins!

# 2. On average, to which airport do flights arrive most early? ----

most_early <- flights %>%
  group_by(dest) %>%
  summarize(delay = mean(arr_delay))

print(most_early)

# there are many NAs in the arr_delay column, use na.rm = TRUE to remove them from mean()

most_early <- flights %>%
  group_by(dest) %>%
  summarize(delay = mean(arr_delay, na.rm = TRUE))

print(most_early)

View(airports)

most_early <- flights %>%
  group_by(dest) %>%
  summarize(delay = mean(arr_delay, na.rm = TRUE)) %>%
  filter(delay == min(delay, na.rm = TRUE)) %>%
  select(dest, delay) %>%
  left_join(airports, by = c("dest" = "faa")) %>%
  select(dest, name, delay)

print(most_early)

# LEX-Blue Grass airport in Lexington, Kentucky is the airport with the earliest
# average arrival time with 22 minutes early

# 3. In which month do flights tend to have the longest delays? ----

flights %>%
  group_by(month) %>%
  summarise(delay = mean(arr_delay, na.rm = TRUE)) %>%
  filter(delay == max(delay)) %>%
  print()

# in month 7 (July) there are longest delays with more than 16 minutes on average

delay_by_month <- flights %>%
  group_by(month) %>%
  summarize(delay = mean(arr_delay, na.rm = TRUE)) %>%
  select(delay) %>%
  mutate(month = month.name) %>% # month.name is a variable built in R
  print()

# Create a plot

ggplot(data = delay_by_month) +
  geom_point(
    mapping = aes(x = delay, y = month),
    color = "blue",
    alpha = .4,
    size = 3
  ) +
  geom_vline(xintercept = 0, size = .25) +
  xlim(c(-20, 20)) +
  scale_y_discrete(limits = rev(month.name)) +
  labs(title = "Average Delay by Month", y = "", x = "Delay (minutes)")
