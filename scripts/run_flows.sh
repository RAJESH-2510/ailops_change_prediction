#!/bin/bash
prefect worker start --pool default-agent-pool --type process &
prefect deploy -f flows/main_flow.py
prefect deploy -f flows/retraining_flow.py
prefect deploy -f flows/real_time_flow.py
