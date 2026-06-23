library(tidycensus)
library(tidyverse)
library(sf)

if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow", repos = "https://cloud.r-project.org")
library(arrow)

# B25035_001E: Median year structure built — pre-computed by Census, clean and simple.
# B25034: Year structure built bins — for modal era, more robust than median in mixed BGs.

STATES <- c(
  "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI",
  "ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN",
  "MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH",
  "OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV",
  "WI","WY"
)

ERA_BREAKS <- function(y) {
  case_when(
    is.na(y)    ~ "unknown",
    y < 1920    ~ "pre-1920",
    y < 1945    ~ "1920s-40s",
    y < 1965    ~ "postwar boom",
    y < 1980    ~ "1965-1979",
    y < 2000    ~ "1980-1999",
    TRUE        ~ "2000+"
  )
}

# B25034 bins for modal era (which decade had the MOST construction in this BG)
BIN_COLS <- c(
  B25034_002E = 2020,  # 2014 or later (ACS 2022)
  B25034_003E = 2010,
  B25034_004E = 2005,
  B25034_005E = 1995,
  B25034_006E = 1985,
  B25034_007E = 1975,
  B25034_008E = 1965,
  B25034_009E = 1955,
  B25034_010E = 1945,
  B25034_011E = 1935
)

modal_year <- function(row_vals, midpoints) {
  # Return midpoint of the bin with the highest housing unit count
  w <- as.numeric(row_vals)
  w[is.na(w) | w < 0] <- 0
  if (sum(w) == 0) return(NA_real_)
  midpoints[which.max(w)]
}

cat("Fetching block group data via tidycensus...\n")

all_bg <- map(STATES, function(state) {
  cat(sprintf("  %s (median year + bins)...\n", state))
  # Get median year built + bin counts + geometry in one call
  get_acs(
    geography = "block group",
    variables = c("B25035_001", names(BIN_COLS)),
    state     = state,
    year      = 2022,
    geometry  = TRUE,
    output    = "wide"
  )
}) |> bind_rows()

cat(sprintf("Fetched %d block groups\n", nrow(all_bg)))

all_bg <- all_bg |>
  rowwise() |>
  mutate(
    median_year_built = B25035_001E,
    modal_year_built  = modal_year(
      c_across(all_of(paste0(names(BIN_COLS)))),
      unname(BIN_COLS)
    ),
    # Use modal year as primary era label — more robust in mixed/gentrifying BGs
    era = ERA_BREAKS(modal_year_built),
    era_median = ERA_BREAKS(median_year_built)
  ) |>
  ungroup() |>
  select(GEOID, median_year_built, modal_year_built, era, era_median, geometry)

cat("Era distribution:\n")
print(count(st_drop_geometry(all_bg), era, sort = TRUE))

out_path <- "/Users/benedictleonardi/Library/CloudStorage/GoogleDrive-benedict.r.leonardi@gmail.com/My Drive/Personal/Random Code/Github_Projects/Street_Type_Divergence/data/processed/block_group_era.parquet"

# Write geometry as WKT so Python/geopandas can read it easily
all_bg_flat <- all_bg |>
  mutate(geometry_wkt = st_as_text(geometry)) |>
  st_drop_geometry()

write_parquet(all_bg_flat, out_path)
cat(sprintf("Saved -> %s\n", out_path))
