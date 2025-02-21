#!/bin/bash

# Assuming same environments as single goal setting 

# Generating 5x5 environments
for i in {2..6}
do
    python3 place_agent_goals.py environments/6x6 $i 1
    python3 place_agent_goals.py environments/7x7 $i 0
    python3 place_agent_goals.py environments/5x5 $i 0
    python3 place_agent_goals.py environments/6x6more_obstacles $i 0 
done


# Generating 6-12 obstacles environments
for i in {6..12}
do
    python3 generate_envs.py 6 $i 25
done
 

TARGET_DIR="environments_init_goal_mg"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR does not exist."
    exit 1
fi

for item in "$TARGET_DIR"/*; do
    python3 generate_samples.py $item
done


