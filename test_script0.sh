experiments=(
  "24 1 0.01 4 1"
  "24 1 0.05 4 1"
)
for exp in "${experiments[@]}"; do
  set -- $exp
  num_node=$1
  device_num=$2
  evec_ratio=$3
  num_head=$4
  temperature=$5

  python test.py --num_node "$num_node" --device "$device_num" --eigenvec_ratio "$evec_ratio" --num_head "$num_head" --temperature "$temperature" 
  
  echo "--------------------------------------"
done