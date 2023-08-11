# LLaMA config
# config_file="./Shell/model_config/Ziya13B_config.json" # change the config file
config_file=$1

base_model=$(jq -r '.base_model' "$config_file")
data_path=$(jq -r '.data_path' "$config_file")
output_dir=$(jq -r '.output_dir' "$config_file")

# echo $base_model

torchrun --nproc_per_node=8 finetune_all.py --base_model ${base_model} --data_path ${data_path} --output_dir ${output_dir}