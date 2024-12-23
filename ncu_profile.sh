server="DGX"

data_type="full"
ncu -o ${data_type}_qwen_long_${server} -f --set detailed python split_mm.py --data_type ${data_type} --is_profile True

data_type="half"
ncu -o ${data_type}_qwen_long_${server} -f --set detailed python split_mm.py --data_type ${data_type} --is_profile True

data_type="bf16"
ncu -o ${data_type}_qwen_long_${server} -f --set detailed python split_mm.py --data_type ${data_type} --is_profile True







# ncu -o ${data_type}_qwen_long_panda -f --section SpeedOfLight_RooflineChart --section LaunchStats python split_mm.py --data_type ${data_type} --is_profile True
