# conda activate multimodal
# HYDRA_FULL_ERROR=1 bash run_scripts/multibench_run.sh


############################# binary classification tasks #############################

# ================================== mosi ==================================
dataset="mosi"
for seed in {42..46};do
      python main_multibench.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            data.data_module.dataset=${dataset} \
            model="infmasking" \
            optim.lr=1e-3 \
            model.model.loss_kwargs.temperature=0.1 \
            model.model.loss_kwargs.mask_lambda=0.25 \
            +model.model.encoder.mask_ratio=0.8 \
            +model.model.encoder.num_mask=5 \
            +exp_name=seed_${seed} 
done

# ================================== humor ==================================
dataset="humor"
for seed in {42..46};do
      python main_multibench.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            data.data_module.dataset=${dataset} \
            model="infmasking" \
            optim.lr=1e-3 \
            model.model.loss_kwargs.temperature=0.05 \
            model.model.loss_kwargs.mask_lambda=2 \
            +model.model.encoder.mask_ratio=0.5 \
            +model.model.encoder.num_mask=4 \
            +model.model.loss_kwargs.weights="[1,1,1,1,5]" \
            +exp_name=seed_${seed} 
done

# ================================== sarcasm ==================================
dataset="sarcasm"
for seed in {42..46};do
      python main_multibench.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            data.data_module.dataset=${dataset} \
            model="infmasking" \
            optim.weight_decay=5e-2 \
            optim.lr=1e-3 \
            model.model.loss_kwargs.temperature=0.05 \
            model.model.loss_kwargs.mask_lambda=1 \
            +model.model.encoder.mask_ratio=0.5 \
            +model.model.encoder.num_mask=5 \
            +exp_name=seed_${seed} 
done

# ================================== mimic ==================================
dataset="mimic"
for seed in {42..46};do
      python main_multibench.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            data.data_module.dataset=${dataset} \
            model="infmasking" \
            model.model.encoder.embed_dim=512 \
            optim.lr=3e-4 \
            model.model.loss_kwargs.temperature=0.2 \
            model.model.loss_kwargs.mask_lambda=1 \
            +model.model.encoder.mask_ratio=0.8 \
            +model.model.encoder.num_mask=6 \
            +model.model.loss_kwargs.weights="[1,1,1,1,5]" \
            +exp_name=seed_${seed} 
done

# ================================== visionandtouch-bin ==================================
dataset="visionandtouch-bin"
for seed in {42..46};do
      python main_multibench.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            data.data_module.dataset=${dataset} \
            model="infmasking" \
            model.model.encoder.embed_dim=512 \
            optim.lr=1e-4 \
            model.model.loss_kwargs.temperature=0.1 \
            model.model.loss_kwargs.mask_lambda=1 \
            +model.model.encoder.mask_ratio=0.7 \
            +model.model.encoder.num_mask=6 \
            +exp_name=seed_${seed} 
done

############################# binary regression tasks #############################

# ================================== visionandtouch ==================================
dataset="visionandtouch"
for seed in {42..46};do
      python main_multibench_reg.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            data.data_module.dataset=${dataset} \
            model="infmasking" \
            model.model.encoder.embed_dim=128 \
            linear_probing.use_sklearn=true \
            linear_probing._target_=evaluation.linear_probe_reg.LinearProbingRegCallback \
            ~linear_probing.fastsearch \
            optim.lr=1e-4 \
            model.model.loss_kwargs.temperature=0.1 \
            model.model.loss_kwargs.mask_lambda=1 \
            +model.model.encoder.mask_ratio=0.5 \
            +model.model.encoder.num_mask=4 \
            +model.model.loss_kwargs.weights="[1,1,1,1,5]" \
            +exp_name=seed_${seed} 
done




