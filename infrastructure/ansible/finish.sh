#!/bin/bash
aws s3 cp /home/ubuntu/out.txt s3://TODO ADD BUCKET NAME HERE/$(date +%s).txt 
/home/ubuntu/cleanup.sh
