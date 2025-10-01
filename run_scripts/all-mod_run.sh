# conda activate multimodal
# HYDRA_FULL_ERROR=1 bash run_scripts/all-mod_run.sh


#  ================================== humor ==================================
# no early stopping
dataset="humor"
for seed in {42..46};do
      python main_multibench_all-mod.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            data.data_module.dataset=${dataset} \
            model="infmasking" \
            model.model.encoder.n_layers=2 \
            optim.lr=1e-3 \
            model.model.loss_kwargs.temperature=0.05 \
            model.model.loss_kwargs.mask_lambda=2 \
            +model.model.encoder.mask_ratio=0.5 \
            +model.model.encoder.num_mask=4 \
            +exp_name=seed_${seed} 
done

# ================================== visionandtouch-bin ==================================
dataset="visionandtouch-bin"
for seed in {42..46};do
      python main_multibench_all-mod.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            data.data_module.dataset=${dataset} \
            model="infmasking" \
            model.model.encoder.n_layers=2 \
            model.model.encoder.embed_dim=512 \
            optim.lr=1e-4 \
            model.model.loss_kwargs.temperature=0.1 \
            model.model.loss_kwargs.mask_lambda=1 \
            +model.model.encoder.mask_ratio=0.8 \
            +model.model.encoder.num_mask=5 \
            +exp_name=seed_${seed} 
done