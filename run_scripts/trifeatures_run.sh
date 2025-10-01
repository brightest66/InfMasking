# conda activate multimodal
# HYDRA_FULL_ERROR=1 bash run_scripts/trifeatures_run.sh

# Set 'true' only for experiments on synergy

for seed in {42..46};do
      # Experiment for Redundancy & Uniqueness
      python main_trifeatures.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_true \
            +data=trifeatures \
            +data.data_module.biased=false \
            +model="infmasking" \
            model.model.encoder.embed_dim=512 \
            optim.lr=3e-4 \
            model.model.loss_kwargs.temperature=0.1 \
            model.model.loss_kwargs.mask_lambda=1 \ 
            +model.model.encoder.mask_ratio=0.7 \
            +model.model.encoder.num_mask=6 \
            mode="train" \
            +exp_name=R-U-seed_${seed}

      # Experiment for Synergy
      python main_trifeatures.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_true \
            +data=trifeatures \
            +data.data_module.biased=true \
            +model="infmasking" \
            model.model.encoder.embed_dim=512 \
            optim.lr=3e-4 \
            model.model.loss_kwargs.temperature=0.1 \
            model.model.loss_kwargs.mask_lambda=1 \
            +model.model.encoder.mask_ratio=0.7 \
            +model.model.encoder.num_mask=6 \
            mode="train" \
            +exp_name=S-seed_${seed} 
done


