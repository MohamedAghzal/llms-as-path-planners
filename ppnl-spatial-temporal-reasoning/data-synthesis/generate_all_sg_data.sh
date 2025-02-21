#!/bin/bash

# Generating 6x6 environments
python3 generate_envs.py 6 1 36

for i in {2..5}
do
    python3 generate_envs.py 6 $i 200
done
python3 place_agent_goals.py environments/6x6 1 1

# Generating 7x7 environments
for i in {1..5}
do
    python3 generate_envs.py 7 $i 25
done
python3 place_agent_goals.py environments/7x7 1 0

# Generating 5x5 environments
for i in {1..5}
do
    python3 generate_envs.py 5 $i 25
done
python3 place_agent_goals.py environments/5x5 1 0

# Generating 6-12 obstacles environments
for i in {6..12}
do
    python3 generate_envs.py 6 $i 25
done
python3 place_agent_goals.py environments/6x6more_obstacles 1 0  

TARGET_DIR="environments_init_goal_sg"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Directory $TARGET_DIR does not exist."
    exit 1
fi

for item in "$TARGET_DIR"/*; do
    python3 generate_samples.py $item
done


