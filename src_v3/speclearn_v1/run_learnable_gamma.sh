#!/bin/bash

# Test learnable gamma parameter for spectral CF
echo "Testing Learnable Gamma Parameter for Spectral CF"
echo "=================================================="

# Test 1: Gowalla with learnable gamma enabled
echo "Test 1: Gowalla with learnable gamma (item view only)"
python main.py \
    --dataset gowalla \
    --filter i \
    --filter_type bernstein \
    --filter_order 8 \
    --i 200 \
    --item_init linear_dec \
    --item_lr 0.01 \
    --item_decay 1e-3 \
    --epochs 20 \
    --loss mse \
    --learnable_gamma \
    --exp_name "gowalla_learnable_gamma" \
    --verbose 1

echo ""
echo "Test 2: Gowalla standard normalization (baseline)"
python main.py \
    --dataset gowalla \
    --filter i \
    --filter_type bernstein \
    --filter_order 8 \
    --i 200 \
    --item_init linear_dec \
    --item_lr 0.01 \
    --item_decay 1e-3 \
    --epochs 20 \
    --loss mse \
    --exp_name "gowalla_standard" \
    --verbose 1

echo ""
echo "Test 3: Yelp2018 with learnable gamma"
python main.py \
    --dataset yelp2018 \
    --filter i \
    --filter_type bernstein \
    --filter_order 8 \
    --i 200 \
    --item_init linear_dec \
    --item_lr 0.01 \
    --item_decay 1e-3 \
    --epochs 20 \
    --loss mse \
    --learnable_gamma \
    --exp_name "yelp2018_learnable_gamma" \
    --verbose 1

echo ""
echo "All tests completed!"