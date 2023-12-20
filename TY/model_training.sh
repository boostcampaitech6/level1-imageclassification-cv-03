cd "$(dirname "$0")"

models=("Conv2Model" "Swin3Model" "EvaModel")
names=("Conv2Model_scaler" "Swin3Model_scaler" "EvaModel_scaler")
resizes=(224 256 448)

for ((i=0; i<${#models[@]}; i++)); do
    model="${models[i]}"
    name="${names[i]}"
    resize="${resizes[i]}"
    python ./code/v2/train.py --model "$model" --model_dir ~/competition_1/model --data_dir ~/data/train/images --lr 1e-4 --resize $resize $resize --epoch 6 --batch_size 128 --name "$name"
done
