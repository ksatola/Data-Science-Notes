
# -------------------------------------
# Exploring Educational Statistics ----
# -------------------------------------

# Load libraries ----

install.packages("ggrepel")

# Load the necessary libraries
library(tidyr) # for data wrangling
library(dplyr) # for data wrangling
library(ggplot2) # for plotting
library(ggrepel) # for plotting
library(scales) # for plotting

# Load data ----

?read.csv

wb_data <- read.csv(
  "data/r/world_bank_data.csv",
  stringsAsFactors = F,
  skip = 4
)

View(wb_data)

# In terms of indicator this data is in long format, 
# in terms of the indicator and year the data is in wide format
# This shape allows comparing indicators across different years

# Visually compare expeditures for 1990 and 2014 ----

# Begin by filtering the rows for the indicator of interest ----
indicator <- "Government expenditure on education, total (% of GDP)"
expenditure_plot_data <- wb_data %>%
  filter(Indicator.Name == indicator)

# Plot the expenditure in 1990 against 2014 using the `ggplot2` package
expenditure_chart <- ggplot(data = expenditure_plot_data) +
  geom_text_repel(
    mapping = aes(x = X1990 / 100, y = X2014 / 100, label = Country.Code)
  ) +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  labs(
    title = indicator, x = "Expenditure 1990",
    y = "Expenditure 2014"
  )

# If you want to visually compare how the expediture across all years varies for a given country
# you would need to reshape your data

# Reshape the data to create a new column for the `year` ----
long_year_data <- wb_data %>%
  gather(
    key = year, # `year` will be the new key column
    value = value, # `value` will be the new value column
    X1960:X # all columns between `X1960` and `X` will be gathered
  )

View(long_year_data)

# Compute fluctuations in an indicator's value over time across all years ----

# Filter the rows for the indicator and country of interest
indicator <- "Government expenditure on education, total (% of GDP)"
spain_plot_data <- long_year_data %>%
  filter(
    Indicator.Name == indicator,
    Country.Code == "ESP" # Spain
  ) %>%
  mutate(year = as.numeric(substr(year, 2, 5))) # remove "X" before each year

# Show the educational expenditure over time ----
chart_title <- paste(indicator, " in Spain")
spain_chart <- ggplot(data = spain_plot_data) +
  geom_line(mapping = aes(x = year, y = value / 100)) +
  scale_y_continuous(labels = percent) +
  labs(title = chart_title, x = "Year", y = "Percent of GDP Expenditure")

# Reshape the data to create columns for each indicator ----
wide_data <- long_year_data %>%
  select(-Indicator.Code) %>% # do not include the `Indicator.Code` column
  spread(
    key = Indicator.Name, # new column names are `Indicator.Name` values
    value = value # populate new columns with values from `value`
  )
# Prepare data and filter for year of interest ----
x_var <- "Literacy rate, adult female (% of females ages 15 and above)"
y_var <- "Unemployment, female (% of female labor force) (modeled ILO estimate)"
lit_plot_data <- wide_data %>%
  mutate(
    lit_percent_2014 = wide_data[, x_var] / 100,
    employ_percent_2014 = wide_data[, y_var] / 100
  ) %>%
  filter(year == "X2014")

# Show the literacy vs. employment rates ----
lit_chart <- ggplot(data = lit_plot_data) +
  geom_point(mapping = aes(x = lit_percent_2014, y = employ_percent_2014)) +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  labs(
    x = x_var,
    y = "Unemployment, female (% of female labor force)",
    title = "Female Literacy Rate versus Female Unemployment Rate"
  )

