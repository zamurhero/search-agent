#!/bin/bash
declare -a cities=(
                    "albanyGA" 
                    "albanyNY" 
                    "albuquerque" 
                    "atlanta" 
                    "augusta" 
                    "austin" 
                    "bakersfield" 
                    "baltimore" 
                    "batonRouge" 
                    "beaumont" 
                    "boise" 
                    "boston" 
                    "buffalo" 
                    "calgary" 
                    "charlotte" 
                    "chattanooga" 
                    "chicago" 
                    "cincinnati" 
                    "cleveland" 
                    "coloradoSprings" 
                    "columbus" 
                    "dallas" 
                    "dayton" 
                    "daytonaBeach" 
                    "denver" 
                    "desMoines" 
                    "elPaso" 
                    "eugene" 
                    "europe" 
                    "ftWorth" 
                    "fresno" 
                    "grandJunction" 
                    "greenBay" 
                    "greensboro" 
                    "houston" 
                    "indianapolis" 
                    "jacksonville" 
                    "japan" 
                    "kansasCity" 
                    "keyWest" 
                    "lafayette" 
                    "lakeCity" 
                    "laredo" 
                    "lasVegas" 
                    "lincoln" 
                    "littleRock" 
                    "losAngeles" 
                    "macon" 
                    "medford" 
                    "memphis" 
                    "mexia" 
                    "mexico" 
                    "miami" 
                    "midland" 
                    "milwaukee" 
                    "minneapolis" 
                    "modesto" 
                    "montreal" 
                    "nashville" 
                    "newHaven" 
                    "newOrleans" 
                    "newYork" 
                    "norfolk" 
                    "oakland" 
                    "oklahomaCity" 
                    "omaha" 
                    "orlando" 
                    "ottawa" 
                    "pensacola" 
                    "philadelphia" 
                    "phoenix" 
                    "pittsburgh" 
                    "pointReyes" 
                    "portland" 
                    "providence" 
                    "provo" 
                    "raleigh" 
                    "redding" 
                    "reno" 
                    "richmond" 
                    "rochester" 
                    "sacramento" 
                    "salem" 
                    "salinas" 
                    "saltLakeCity" 
                    "sanAntonio" 
                    "sanDiego" 
                    "sanFrancisco" 
                    "sanJose" 
                    "sanLuisObispo" 
                    "santaFe" 
                    "saultSteMarie" 
                    "savannah" 
                    "seattle" 
                    "stLouis" 
                    "stamford" 
                    "stockton" 
                    "tallahassee" 
                    "tampa" 
                    "thunderBay" 
                    "toledo" 
                    "toronto" 
                    "tucson" 
                    "tulsa" 
                    "uk1" 
                    "uk2" 
                    "vancouver" 
                    "washington" 
                    "westPalmBeach" 
                    "wichita" 
                    "winnipeg" 
                    "yuma" 
                   )
declare -a algos=("DFS" "A*" "RBFS")
for i in "${cities[@]}"
do
    for j in "${cities[@]}"
    do
        python3 Search.py DFS 0 "$i" "$j" >> dfs_stats.csv
    done
done