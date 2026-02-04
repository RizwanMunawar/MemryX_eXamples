#!/bin/bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -P ../../assets/ImageNet2012_valdata
mkdir ../../assets/ImageNet2012_valdata/images
tar xvf ../../assets/ImageNet2012_valdata/ILSVRC2012_img_val.tar -C ../../assets/ImageNet2012_valdata/images
