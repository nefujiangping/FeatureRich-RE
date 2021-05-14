CUDA_VISIBLE_DEVICES=0 python main.py \
--data_dir ./data \
--meta_dir ./meta \
--transformer_type bert \
--model_name_or_path ./sciBERT \
--test_file test.json \
--test_batch_size 16 \
--max_seq_length 50 \
--num_labels 8 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 8 \
--mode test \
--load_path checkpoint/FeatureRich-RE-baseline.pt
