# EXPLORING CHANGES TO THE CITY OF SEATTLE ----

# Evaluate the claim that “The City of Seattle is changing” 
# (in large part due to the growing technology industry) 
# by analyzing construction projects as documented 
# through building permit data14 downloaded 
# from the City of Seattle’s open data program.

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

