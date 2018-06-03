#!/bin/bash

# python3 main.py --model LazyNet --epochs 50
# pythyon3 main.py --model BoringNet --epochs 50
# python3 main.py --model CoolNet_v1 --epochs 50

# python3 main.py --model CoolNet_v2 --epochs 30
# python3 main.py --model CoolNet_v2_3 --epochs 30
# python3 main.py --model CoolNet_v2_9 --epochs 30
# python3 main.py --model CoolNet_v3 --epochs 30
# python3 main.py --model CoolNet_v4 --epochs 30
# python3 main.py --model CoolNet_v5 --epochs 30
# python3 main.py --model CoolNet_v6 --epochs 30
# python3 main.py --model CoolNet_v7 --epochs 30
# python3 main.py --model CoolNet_v8 --epochs 30
# python3 main.py --model CoolNet_v9 --epochs 30

# python3 main.py --model CoolNet_v3 --epochs 50
# python3 main.py --model CoolNet_v4 --epochs 50
# python3 main.py --model CoolNet_v7 --epochs 50
# python3 main.py --model CoolNet_v10 --epochs 50

# Try using three different values for batch size. How do these values affect training and why?
# Default is 4
# python3 main.py --model CoolNet --epochs 50 --batchSize 20
# python3 main.py --model CoolNet --epochs 50 --batchSize 10
# python3 main.py --model CoolNet --epochs 50 --batchSize 5
# python3 main.py --model CoolNet_v7 --epochs 50 --batchSize 20
# python3 main.py --model CoolNet_v7 --epochs 50 --batchSize 10
# python3 main.py --model CoolNet_v7 --epochs 50 --batchSize 5

# Try to train your model with different learning rates and plot the training accuracy, test accuracy and loss and compare the training progress for learning rates = 10, 0.1, 0.01, 0.0001
# Analyze the results and choose the best one. Why did you choose this value?
# python3 main.py --lr 10 --model CoolNet --epochs 50
# python3 main.py --lr .1 --model CoolNet --epochs 50
# python3 main.py --lr .01 --model CoolNet --epochs 50
# python3 main.py --lr .0001 --model CoolNet --epochs 50
# python3 main.py --lr 10 --model CoolNet_v7 --epochs 50
# python3 main.py --lr .1 --model CoolNet_v7 --epochs 50
# python3 main.py --lr .01 --model CoolNet_v7 --epochs 50
# python3 main.py --lr .0001 --model CoolNet_v7 --epochs 50

# python3 main.py --model CoolNet --epochs 150
# python3 main.py --model CoolNet_v7 --epochs 150

# python3 main.py --epochs 200 --model CoolNet --lr 0.01
# python3 main.py --epochs 200 --model CoolNet_v7 --lr 0.01

# python3 main.py --epochs 200 --model CoolNet --lr 0.01
# python3 main.py --epochs 200 --model CoolNet_v7 --lr 0.01

# python3 main.py --epochs 200 --model CoolNet --lr 0.01
python3 main.py --epochs 200 --model CoolNet_v7 --lr 0.01