# conda activate multimodal
# HYDRA_FULL_ERROR=1 bash run_scripts/test.sh

################################## test single model.ckpt ##################################
dataset="mimic" # Adjust the parameters according to the dataset
ckpt_path="'/path/to/your/best_checkpoint_on_validation.ckpt'"
python main_multibench.py \
    trainer.strategy=ddp_find_unused_parameters_false \
    data.data_module.dataset=${dataset} \
    model="infmasking" \
    model.model.encoder.embed_dim=512 \
    +model.model.encoder.num_mask=${num_mask} \
    ...  # other hyperparameters you used during training \
    mode="test" \
    +ckpt_path=${ckpt_path} 


################################## test multiple model.ckpt in a directory ##################################
# Set the directory to search for ckpt files
SEARCH_DIR="/path/to/your/directory"

# check if the directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory $SEARCH_DIR does not exist!"
    exit 1
fi

# Find all ckpt files (recursively search for checkpoints in seed_* directories)
CKPT_FILES=$(find "$SEARCH_DIR" -type f -path "*/seed_*/logs/*/checkpoints/*.ckpt")

# Check if any ckpt files were found
if [ -z "$CKPT_FILES" ]; then
    echo "No ckpt files found in $SEARCH_DIR!"
    exit 1
fi

# Iterate over all found ckpt files and execute the test command
for ckpt_path in $CKPT_FILES; do
    echo "========================================"
    echo "Found ckpt file: $ckpt_path"

    # Extract seed value from ckpt path (get number from seed_* directory name)
    seed=$(echo "$ckpt_path" | grep -oP 'seed_\K\d+')

    # Check if seed was successfully extracted
    if [ -z "$seed" ]; then
        echo "Warning: Could not extract seed value from path, skipping this file. "
        continue
    fi
    
    echo "Extracted seed: $seed"
    echo "Starting test command..."
    echo "========================================"

    # Execute test command
    # --------------------------------------------------------

    # your_command_here, please adjust the parameters according to the dataset

    # --------------------------------------------------------

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "========================================"
        echo "seed=$seed test executed successfully!"
    else
        echo "========================================"
        echo "seed=$seed test failed!"
    fi
    
    echo "\n"
done

echo "All tests completed."