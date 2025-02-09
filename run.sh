#!/bin/bash

# Specify the Python interpreter (which can be modified to your Python path)
PYTHON_CMD="python"

# Setting the Default Parameters
QUESTION_TYPE="choice"  

# Parsing command line arguments
if [[ $# -ge 1 ]]; then
    QUESTION_TYPE="$1"
fi

echo "Running main_MARS.py with question type: $QUESTION_TYPE"

# Run the Python script and pass the topic type parameter
$PYTHON_CMD main_MARS.py "$QUESTION_TYPE"


# run.sh short_answer
# run.sh