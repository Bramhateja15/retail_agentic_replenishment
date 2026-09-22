# Retail Agentic Replenishment — Data Foundation

MSc Business Analytics Project (BUSI1783), University of Greenwich.

## Current scope
This repository currently contains ONLY the data foundation:
raw import, cleaning, preparation and date-based integration of
FreshRetailNet-50K (primary operational retail dataset) and GDELT
(external daily contextual dataset). No forecasting, modelling,
simulation, or agentic components exist yet.

## Pipeline layers
raw/ -> cleaned/ -> processed/ (integration happens in processed/)

Raw data is never modified in place. Each layer is produced by a
dedicated script under scripts/.

## Setup
See config/paths.py and requirements.txt.