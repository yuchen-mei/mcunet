#!/bin/bash

for dir in "./imagenet/val" "./vww-s256/val"; do
    if [ ! -d "$dir" ]; then
        echo "Error: $dir does not exist."
        exit 1
    fi
done


echo -e "\nTesting ImageNet0"
python3 eval_torch.py --net_id mcunet-in0 --dataset imagenet --data-dir ./imagenet/val
python3 eval_torch.py --net_id mcunet-in0 --dataset imagenet --data-dir ./imagenet/val --bfloat16

echo -e "\nTesting ImageNet1"
python3 eval_torch.py --net_id mcunet-in1 --dataset imagenet --data-dir ./imagenet/val
python3 eval_torch.py --net_id mcunet-in1 --dataset imagenet --data-dir ./imagenet/val --bfloat16

echo -e "\nTesting ImageNet2"
python3 eval_torch.py --net_id mcunet-in2 --dataset imagenet --data-dir ./imagenet/val
python3 eval_torch.py --net_id mcunet-in2 --dataset imagenet --data-dir ./imagenet/val --bfloat16

echo -e "\nTesting ImageNet3"
python3 eval_torch.py --net_id mcunet-in3 --dataset imagenet --data-dir ./imagenet/val
python3 eval_torch.py --net_id mcunet-in3 --dataset imagenet --data-dir ./imagenet/val --bfloat16

echo -e "\nTesting ImageNet4"
python3 eval_torch.py --net_id mcunet-in4 --dataset imagenet --data-dir ./imagenet/val
python3 eval_torch.py --net_id mcunet-in4 --dataset imagenet --data-dir ./imagenet/val --bfloat16

echo -e "\nTesting VWW0"
python3 eval_torch.py --net_id mcunet-vww0 --dataset vww --data-dir ./vww-s256/val
python3 eval_torch.py --net_id mcunet-vww0 --dataset vww --data-dir ./vww-s256/val --bfloat16

echo -e "\nTesting VWW1"
python3 eval_torch.py --net_id mcunet-vww1 --dataset vww --data-dir ./vww-s256/val
python3 eval_torch.py --net_id mcunet-vww1 --dataset vww --data-dir ./vww-s256/val --bfloat16

echo -e "\nTesting VWW2"
python3 eval_torch.py --net_id mcunet-vww2 --dataset vww --data-dir ./vww-s256/val
python3 eval_torch.py --net_id mcunet-vww2 --dataset vww --data-dir ./vww-s256/val --bfloat16

# int8
# python3 eval_tflite.py --net_id mcunet-in4 --dataset imagenet --data-dir ./imagenet/val

# single image
python3 eval_single.py --net_id mcunet-in4 --fuse-bn --bfloat16
python3 eval_single.py --net_id mcunet-in4 --bfloat16
