#!/bin/bash
DATE=$(date "+%Y-%m-%d-%H%M")
raspistill -o /home/blud/test/$DATE.jpg
