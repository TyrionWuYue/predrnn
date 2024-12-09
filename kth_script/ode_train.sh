export CUDA_VISIBLE_DEVICES=4
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name action \
    --train_data_paths dataset/kth_action \
    --valid_data_paths dataset/kth_action \
    --save_dir checkpoints/kth_DDGODE \
    --gen_frm_dir results/kth_DDGODE \
    --model_name DDGODE \
    --visual 0 \
    --reverse_input 0 \
    --img_width 128 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128 \
    --patch_size 2 \
    --layer_norm 0 \
    --decouple_beta 0.01 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 5000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2000 \
    --lr 0.0001 \
    --batch_size 4 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 1000 \
#    --pretrained_model ./checkpoints/kth_predrnn_v2/kth_model.ckpt